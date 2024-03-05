
#include "gmxpre.h"

#include "ewald.h"
//#include "ewald.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <algorithm> 

#include "gromacs/ewald/ewald_utils.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/gmxcomplex.h"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h" 
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"

#include<iostream>
#include<stdio.h>
#include <immintrin.h>
#include <mkl.h> 

using cvec = std::array<t_complex, DIM>;//t_complex   ͵        DIM  

gmx_ewald_tab_t::gmx_ewald_tab_t(const t_inputrec& ir, FILE* fp)
{
    if (fp) 
    {
        fprintf(fp, "Will do ordinary reciprocal space Ewald sum.\n");
        fprintf(fp, "This package is running.\n");
    }

    P = ir.userint1; 
    Step = 0;
    snew(K, P);
    snew(Comm_All, P);

    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_max = ir.userint2;//   ж  ٸ  
    
    kz_max = 20;

    if ((size < size_max)|| (size_max==0))
    {
        std::cout << "The size_max is changed" << std::endl;
        size_max = 1;
    }

    if (Step == 0)
        srand(2003+me);//    Ҫ   Ȼ    CPU         Ƶ     
}

gmx_ewald_tab_t::~gmx_ewald_tab_t() = default;

//! Calculates wave vectors.
static void calc_lll(const rvec box, rvec lll)
{
    lll[XX] = 2.0 * M_PI / box[XX];
    lll[YY] = 2.0 * M_PI / box[YY];
    lll[ZZ] = 2.0 * M_PI / box[ZZ];
} 

inline real Prob(real kx, real ky, real kz, real alpha, real S)
{
    return exp(-(kx * kx + ky * ky + kz * kz) / (4 * alpha)) / S;
}

static real MH_D(real m, real coef)
{
    real a;
    if (m == 0 || fabs(m) < 1.0e-13)
    {
        a = erf(0.5 * coef);
    }
    else
    {
        a = 0.5 * ( erf((abs(m) + 0.5) * coef) - erf((abs(m) - 0.5) * coef) );
    }
    return a;
}

//         
static double randn_box_muller_linear_congruential(const double Mean, const double SquareMargin)
{
    const double epsilon = 1.17549e-038;
    const double two_pi = 2.0 * 3.14159265358979323846;
    static double z0, z1;
    double u1, u2;
    do {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * SquareMargin + Mean;
}

real do_ewald(bool                           havePbcXY2Walls,
              real                           wallEwaldZfac,
              real                           epsilonR,
              FreeEnergyPerturbationType     freeEnergyPerturbationType,
              gmx::ArrayRef<const gmx::RVec> coords,//λ      
              gmx::ArrayRef<gmx::RVec>       forces,//        
              gmx::ArrayRef<const real>      chargeA,
              gmx::ArrayRef<const real>      chargeB,
              const matrix                   box,
              const t_commrec*               commrec,
              int                            natoms,
              matrix                         lrvir,
              real                           ewaldcoeff,
              real                           lambda,
              real*                          dvdlambda,
              gmx_ewald_tab_t*               et)
{
    //MPI_Barrier(MPI_COMM_WORLD);
    //double t1, t2;
    //t1 = MPI_Wtime();

    real   factor = -1.0 / (4 * ewaldcoeff * ewaldcoeff);
    real   energy_AB[2], energy;
    real   alpha, box_coe_x, box_coe_y, box_coe_z, Sx, Sy, Sz, V, scale;
    real   mold_x, mold_y, mold_z, mnew_x, mnew_y, mnew_z, pup, pdown,qup, qdown, acce, rand_num;
    real   KK[et->P][3]; 
    rvec   lll;

    bool   bFreeEnergy;

    //real x[int(ceil((natoms + 0.0) / 16.0)) * 16][3]; real f[int(ceil((natoms + 0.0) / 16.0)) * 16][3];  real q[int(ceil((natoms + 0.0) / 16.0)) * 16];

    /* Scale box with Ewald wall factor */
    matrix          scaledBox;
    EwaldBoxZScaler boxScaler(havePbcXY2Walls, wallEwaldZfac);
    boxScaler.scaleBox(box, scaledBox);

    rvec boxDiag;
    for (int i = 0; (i < DIM); i++)
    {
        boxDiag[i] = scaledBox[i][i];
    }

    bFreeEnergy = (freeEnergyPerturbationType != FreeEnergyPerturbationType::No);
    clear_mat(lrvir);
    calc_lll(boxDiag, lll);

    //        
    alpha = ewaldcoeff * ewaldcoeff;

    //   Ǳ  θ  º ĺ   
    box_coe_x = alpha * boxDiag[0] * boxDiag[0];
    box_coe_y = alpha * boxDiag[1] * boxDiag[1];
    box_coe_z = alpha * boxDiag[2] * boxDiag[2];
    Sx = sqrt(box_coe_x / M_PI) * (1 + 2 * exp(-box_coe_x) + 2 * exp(-box_coe_x * 4.0));
    Sy = sqrt(box_coe_y / M_PI) * (1 + 2 * exp(-box_coe_y) + 2 * exp(-box_coe_y * 4.0));
    Sz = sqrt(box_coe_z / M_PI) * (1 + 2 * exp(-box_coe_z) + 2 * exp(-box_coe_z * 4.0));
    et->S = Sx * Sy * Sz - 1;
    V = boxDiag[0] * boxDiag[1] * boxDiag[2];
    
    //修正S的值
    real smalls1z = 0.00;
    for (int kz = -et->kz_max; kz <= et->kz_max; kz++)
    {
        smalls1z = smalls1z + std::exp( - M_PI * M_PI * kz * kz / (alpha * boxDiag[2] * boxDiag[2]));
    }
    smalls1z = smalls1z - 1;
    et->S = et->S - smalls1z;

    /* 1/(Vol*e0) */
    real scaleRecip = 4.0 * M_PI / (boxDiag[XX] * boxDiag[YY] * boxDiag[ZZ]) * (gmx::c_one4PiEps0 / epsilonR);


    int this_rank = (et->Step % et->size_max);
    if ( (et->Step % et->size_max) == 0 )
    {
        //  ¼ ³   ʱ ̵Ĳ    ÿ  rank  Ҫ  
        et->S_memory = et->S;
        et->Box_memory[0] = boxDiag[0];
        et->Box_memory[1] = boxDiag[1];
        et->Box_memory[2] = boxDiag[2];

        int index = 0;
        real coef_x_sample = sqrt(box_coe_x / (2 * M_PI * M_PI));
        real coef_y_sample = sqrt(box_coe_y / (2 * M_PI * M_PI));
        real coef_z_sample = sqrt(box_coe_z / (2 * M_PI * M_PI));

        real coef_x_MHD = 1.0/(sqrt(box_coe_x / ( M_PI * M_PI)));
        real coef_y_MHD = 1.0/(sqrt(box_coe_y / (M_PI * M_PI)));
        real coef_z_MHD = 1.0/(sqrt(box_coe_z / (M_PI * M_PI)));

        if ((et->me < et->size_max)) {
            int mx[5 * et->P], my[5 * et->P], mz[5 * et->P];

            do
            {
                mx[0] = round(randn_box_muller_linear_congruential(0, coef_x_sample));
                my[0] = round(randn_box_muller_linear_congruential(0, coef_y_sample));
                mz[0] = round(randn_box_muller_linear_congruential(0, coef_z_sample));
            } while (mx[0] == 0 && my[0] == 0 && abs(mz[0]) <= et->kz_max);
            
            for (int i = 1; i < 5 * et->P; i++)
            {
                mold_x = mx[i - 1]+0.00;
                mold_y = my[i - 1]+0.00;
                mold_z = mz[i - 1]+0.00;
                mnew_x = round(randn_box_muller_linear_congruential(0, coef_x_sample))+0.00;
                mnew_y = round(randn_box_muller_linear_congruential(0, coef_y_sample))+0.00;
                mnew_z = round(randn_box_muller_linear_congruential(0, coef_z_sample))+0.00;

                pup = Prob(mnew_x * lll[0], mnew_y * lll[1], mnew_z * lll[2], alpha, et->S);
                pdown = Prob(mold_x * lll[0], mold_y * lll[1], mold_z * lll[2], alpha, et->S);

                qup = MH_D(mold_x, coef_x_MHD)* MH_D(mold_y, coef_y_MHD)* MH_D(mold_z, coef_z_MHD);//   кܴ      ռ 
                qdown = MH_D(mnew_x, coef_x_MHD)* MH_D(mnew_y, coef_y_MHD)* MH_D(mnew_z, coef_z_MHD);

                acce = pup * qup / (pdown * qdown) > 1.0 ? 1.0 : pup * qup / (pdown * qdown);
                rand_num = (rand() % 100000 + 0.00) / 100000.0;
                if (rand_num < acce)
                {
                    mx[i] = mnew_x;
                    my[i] = mnew_y;
                    mz[i] = mnew_z;
                    index++;
                }
                else
                {
                    mx[i] = mold_x;
                    my[i] = mold_y;
                    mz[i] = mold_z;
                }

                if (mx[i] == 0 && my[i] == 0 && abs(mz[i]) <= et->kz_max)
                {
                    i = i - 1;
                }
            }
            for (int i = 0; i < et->P; i++)
            {
                et->K[i][0] = mx[5 * i + 4] + 0.00;
                et->K[i][1] = my[5 * i + 4] + 0.00;
                et->K[i][2] = mz[5 * i + 4] + 0.00; 

                //std::cout << "This is rank " << et->me << "     " << et->K[i][0] << "   " << et->K[i][1] << "   " << et->K[i][2] << std::endl;
            }
        }

        //if (et->me <= 2)
        //{
        //    std::cout << "The acceptance rate is " << (index + 0.00) / (et->P + 0.00) << std::endl;
        //    std::cout << "This is rank "<<et->me<<"     "<<et->K[0][0] << "   " << et->K[0][1] << "   " << et->K[0][2] << std::endl;
        //}
    }

    if (et->me == this_rank)
    {
        // Ѿ  ˺       
        for (int i = 0; i < et->P; i++)
        {
            KK[i][0] = et->K[i][0] * lll[0];
            KK[i][1] = et->K[i][1] * lll[1];
            KK[i][2] = et->K[i][2] * lll[2];
        }
    }
    
    //std::cout << "This is rank "<<et->me<<"    "<<et->K[1][0] << "    " << et->K[1][1] << "    " << et->K[1][2] << "    "  << et->P << std::endl;

    //t2 = MPI_Wtime();
    //if (et->me == 0)std::cout << "Time of the I part is " << (t2 - t1) * 1000 << "  ms" << std::endl;

    MPI_Bcast((real*)KK, 3 * et->P, MPI_FLOAT, this_rank, MPI_COMM_WORLD);
    
    //std::cout << "This is rank " << et->me << "    "<<KK[1][0] << "  " << KK[1][1] << "  " << KK[1][2] << "   " << et->P << std::endl;

    et->Step++;

    //t2 = MPI_Wtime();
    //if (et->me == 0)std::cout << "Time of the I.5 part is " << (t2 - t1) * 1000 << "  ms" << std::endl;

    //std::cout << "The paras are "<<epsilonR <<"    "<< ewaldcoeff<< std::endl;

    int natom_new = int(ceil((natoms + 0.0) / 16.0)) * 16;
    real X[natom_new],Y[natom_new],Z[natom_new]; 
    real F[natom_new][3];  
    real Q[natom_new];
    
    int P_new = int(ceil((et->P + 0.0) / 16.0)) * 16;
    real KKKx[P_new], KKKy[P_new], KKKz[P_new], KKSquare[P_new];
    real Rho[et->P][2];
    real Rho_Cos[P_new], Rho_Sin[P_new];
    real memoryfactor[P_new];

    for (int i = 0; i < natom_new; i++)
    {
        if (i < natoms) {
            X[i] = coords[i][0];
            Y[i] = coords[i][1];
            Z[i] = coords[i][2];
            F[i][0] = 0.00; 
            F[i][1] = 0.00; 
            F[i][2] = 0.00;
            Q[i] = chargeA[i];
        }
        else
        {
            X[i] = 0.00;
            Y[i] = 0.00;
            Z[i] = 0.00;
            F[i][0] = 0.00;
            F[i][1] = 0.00;
            F[i][2] = 0.00;
            Q[i] = 0.00;
        }
    }

    real lll_mem[3] = { boxDiag[0] / et->Box_memory[0], boxDiag[1] / et->Box_memory[1], boxDiag[2] / et->Box_memory[2] };
    real k2_Mem,k2;

    for (int i = 0; i < P_new; i++)
    {
        if (i < et->P) {
            KKKx[i] = KK[i][0];    KKKy[i] = KK[i][1];    KKKz[i] = KK[i][2];
            KKSquare[i] = KKKx[i] * KKKx[i] + KKKy[i] * KKKy[i] + KKKz[i] * KKKz[i];

            k2_Mem = (KKKx[i] * lll_mem[0]) * (KKKx[i] * lll_mem[0]) + (KKKy[i] * lll_mem[1]) * (KKKy[i] * lll_mem[1]) + (KKKz[i] * lll_mem[2]) * (KKKz[i] * lll_mem[2]);
            k2_Mem = k2_Mem / (4 * ewaldcoeff * ewaldcoeff);
            k2 = KKSquare[i] / (4 * ewaldcoeff * ewaldcoeff);
            memoryfactor[i] = std::exp(-k2 + k2_Mem);      
        }
        else if (i >= et->P)
        {
            KKKx[i] = 0.00;    KKKy[i] = 0.00;    KKKz[i] = 0.00;
            KKSquare[i] = 0.00;
            
            memoryfactor[i] = 0.00;
        }
    }
   
    //double t3 = MPI_Wtime();
    //if (et->me == 0)std::cout << "Time of the II-1 part is " << (t3 - t2) * 1000 << "  ms" << std::endl;

    real addenergy = 0.0;

    gmx::ArrayRef<const real> charge;
    for (int q = 0; q < (bFreeEnergy ? 2 : 1); q++)
    {
        if (!bFreeEnergy)
        {
            charge = chargeA;
            scale  = 1.0;
        }
        else if (q == 0)
        {
            charge = chargeA;
            scale  = 1.0 - lambda; 
        }
        else
        {
            charge = chargeB;
            scale  = lambda;
        }

        energy_AB[q] = 0;

        __m512 Kx, Ky, Kz, Real, Imag, XX0, YY0, ZZ0, QQ0, moment, Sin, Cos;
        const int    SIMD = 16;

        for (int i = 0; i < et->P; i += SIMD)
        {          
            Real = Imag = _mm512_setzero_ps();
            Kx = _mm512_load_ps(&KKKx[i]);
            Ky = _mm512_load_ps(&KKKy[i]);
            Kz = _mm512_load_ps(&KKKz[i]);

            for (int j = 0; j < natoms; j++)
            {
                XX0 = _mm512_set1_ps(coords[j][0]);
                YY0 = _mm512_set1_ps(coords[j][1]);
                ZZ0 = _mm512_set1_ps(coords[j][2]);
                QQ0 = _mm512_set1_ps(charge[j]);
                moment = Kx * XX0 + Ky * YY0 + Kz * ZZ0;
                Sin = _mm512_sincos_ps(&Cos, moment);
                Real = Real + QQ0 * Cos;
                Imag = Imag + QQ0 * Sin;
            }
            _mm512_store_ps(&Rho_Cos[i], Real);
            _mm512_store_ps(&Rho_Sin[i], Imag);
        }
        for (int i = 0; i < et->P; i++)
        {
            Rho[i][0] = Rho_Cos[i];
            Rho[i][1] = Rho_Sin[i];
        }

        MPI_Allreduce((float*)Rho, (float*)et->Comm_All, 2 * et->P, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        //double t4 = MPI_Wtime();
        //if (et->me == 0)std::cout << "Time of the II.2 part is " << (t4 - t3) * 1000 << "  ms" << std::endl;

        __m512 scaleRecip = _mm512_set1_ps( -4 * M_PI * (et->S_memory / (et->P + 0.00)) * (gmx::c_one4PiEps0 / epsilonR) / (V + 0.00));// ˴  ޸ 

        __m512 Fx, Fy, Fz, ksquare, midterm; //__m512 KX_Mem, KY_Mem, KZ_Mem, K2_Mem; real k2_Mem;

        __m512 MemoryFactor;

        real   XXX[SIMD], YYY[SIMD], ZZZ[SIMD];

        for (int i = 0; i < natoms; i += SIMD)
        {
            Fx = Fy = Fz = _mm512_set1_ps(0.00);

            XX0 = _mm512_load_ps(&X[i]);
            YY0 = _mm512_load_ps(&Y[i]); 
            ZZ0 = _mm512_load_ps(&Z[i]);
            QQ0 = _mm512_load_ps(&Q[i]);

            for (int j = 0; j < et->P; j++)
            {
                Kx = _mm512_set1_ps(KKKx[j]);
                Ky = _mm512_set1_ps(KKKy[j]);
                Kz = _mm512_set1_ps(KKKz[j]);

                moment = -(Kx * XX0 + Ky * YY0 + Kz * ZZ0);
                ksquare = _mm512_set1_ps(KKSquare[j]);

                Sin = _mm512_sincos_ps(&Cos, moment);
                Real = _mm512_set1_ps(et->Comm_All[j].re);
                Imag = _mm512_set1_ps(et->Comm_All[j].im);

                MemoryFactor = _mm512_set1_ps(memoryfactor[j]);

                midterm = QQ0 * (Cos * Imag + Sin * Real) * scaleRecip * MemoryFactor / (ksquare);
                Fx = Fx + Kx * midterm;
                Fy = Fy + Ky * midterm;
                Fz = Fz + Kz * midterm;
            }
            
            _mm512_store_ps(XXX, Fx); 
            _mm512_store_ps(YYY, Fy);
            _mm512_store_ps(ZZZ, Fz);
            for (int j = 0; j < SIMD; j++)
            {
                    F[i + j][0] = F[i + j][0] + XXX[j];
                    F[i + j][1] = F[i + j][1] + YYY[j]; 
                    F[i + j][2] = F[i + j][2] + ZZZ[j];
            }
        }

        for (int i = 0; i < natoms; i++)
        {
            forces[i][0] += F[i][0];
            forces[i][1] += F[i][1];
            forces[i][2] += F[i][2];
        }

        //printf("Begin SPMF");

        /************       SPMF部分的计算       **************/
        real k_rhok[2 * et->kz_max + 1];
        for (int i = -et->kz_max; i <= et->kz_max; i++)
        {
            k_rhok[i + et->kz_max] = 2.0 * M_PI * i / boxDiag[2];
        }
        real constFterm = -4 * M_PI * (gmx::c_one4PiEps0 / epsilonR) / (V);
        real rhok[2 * et->kz_max + 1][2], rholocal[2 * et->kz_max + 1][2];

        for (int k = -et->kz_max; k <= et->kz_max; k++)
        {
            rholocal[k + et->kz_max][0] = 0.0;
            rholocal[k + et->kz_max][1] = 0.0;
            rhok[k + et->kz_max][0] = 0.0;
            rhok[k + et->kz_max][1] = 0.0;
            for (int i = 0; i < natoms; i++)
            {
                rholocal[k + et->kz_max][0] = rholocal[k + et->kz_max][0] + Q[i] * cosf(k_rhok[k + et->kz_max] * Z[i]);
                rholocal[k + et->kz_max][1] = rholocal[k + et->kz_max][1] + Q[i] * sinf(k_rhok[k + et->kz_max] * Z[i]);
            }
        }
 
        MPI_Allreduce(&rholocal[0][0], &rhok[0][0], 2 * (2 * et->kz_max + 1), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        real imageF[natoms][2 * et->kz_max + 1];
        real ksqrt[2 * et->kz_max + 1], ktermF[2 * et->kz_max + 1];//定义kz平方
        for (int k = -et->kz_max; k <= et->kz_max; k++)
        {
            ksqrt[k + et->kz_max] = k_rhok[k + et->kz_max] * k_rhok[k + et->kz_max];
            if (k != 0)
            {
                ktermF[k + et->kz_max] = k_rhok[k + et->kz_max] / ksqrt[k + et->kz_max] * std::exp(-ksqrt[k + et->kz_max] / (4.0 * alpha));
            }
            else
            {
                ktermF[k + et->kz_max] = 0.00;
            }
            for (int i = 0; i < natoms; i++)
            {
                imageF[i][k + et->kz_max] = (cosf(k_rhok[k + et->kz_max] * Z[i]) * rhok[k + et->kz_max][1] - sinf(k_rhok[k + et->kz_max] * Z[i]) * rhok[k + et->kz_max][0]);
            }
        }

        real smallkF[natoms];

        for (int i = 0; i < natoms; i++)
        {
            smallkF[i] = 0.0;
        }
        for (int i = 0; i < natoms; i++)
        {
            for (int k = -et->kz_max; k <= et->kz_max; k++)
            {
                smallkF[i] = smallkF[i] + constFterm * Q[i] * ktermF[k + et->kz_max] * imageF[i][k + et->kz_max];
            }
        }
        for (int i = 0; i < natoms; i++)
        {
            forces[i][2] += smallkF[i];
        }

        //printf("The force is %lf, %lf, %lf\n", forces[0][0], forces[0][1], forces[0][2]);
        /*************       SPMF部分计算结束      ***************/

        //double t5 = MPI_Wtime();
        //if (et->me == 0)std::cout << "Time of the II.3 part is " << (t5 - t4) * 1000 << "  ms" << std::endl;

        real scaleVirial = -M_PI * (et->S_memory / (et->P + 0.00)) * (gmx::c_one4PiEps0 / epsilonR) / (V + 0.00);
        //real scaleVirial_npt = M_PI * (et->S_vir_memory / (et->P + 0.00)) * (gmx::c_one4PiEps0 / epsilonR) / (V + 0.00);
        real scaleNew = - M_PI / (boxDiag[XX] * boxDiag[YY] * boxDiag[ZZ]) * gmx::c_one4PiEps0 / epsilonR;

        for (int i = 0; i < et->P; i++)
        {
            real k1 = KKKx[i], k2 = KKKy[i], k3 = KKKz[i];
            real k_square = KKSquare[i];

            real ak = 1.0 / k_square;
            real akv = 2.0 * (1.0 / k_square - factor);
            energy_AB[q] -= 0.5 * ak * (et->Comm_All[i].re * et->Comm_All[i].re + et->Comm_All[i].im * et->Comm_All[i].im);
            real tmp = scaleVirial * scale * (1.0 / k_square) * memoryfactor[i] * (et->Comm_All[i].re * et->Comm_All[i].re + et->Comm_All[i].im * et->Comm_All[i].im);

            //Ewald因为只算了一半 所以要乘2系数 我们这里不用 
            lrvir[XX][XX] += tmp * (1 - akv * k1 * k1);
            lrvir[XX][YY] += tmp * (-akv * k1 * k2);
            lrvir[XX][ZZ] += tmp * (-akv * k1 * k3);
            lrvir[YY][YY] += tmp * (1 - akv * k2 * k2);
            lrvir[YY][ZZ] += tmp * (-akv * k2 * k3);
            lrvir[ZZ][ZZ] += tmp * (1 - akv * k3 * k3); 

            /*
            real k1 = KKKx[i], k2 = KKKy[i], k3 = KKKz[i];
            real k_square = KKSquare[i];

            real k1_npt = KKKx_npt[i], k2_npt = KKKy_npt[i], k3_npt = KKKz_npt[i];
            real k_square_npt = KKSquare_npt[i];

            real ak = 1.0 / k_square;
            energy_AB[q] -= 0.5 * ak * (et->Comm_All[i].re * et->Comm_All[i].re + et->Comm_All[i].im * et->Comm_All[i].im);

            real tmp = scaleVirial * scale * (1.0 / k_square) * memoryfactor[i] * (et->Comm_All[i].re * et->Comm_All[i].re + et->Comm_All[i].im * et->Comm_All[i].im);
            real tmp_npt = scaleVirial_npt * scale * (1.0 / (k_square_npt * k_square_npt)) * memoryfactor_npt[i] * (et->Comm_All_npt[i].re * et->Comm_All_npt[i].re + et->Comm_All_npt[i].im * et->Comm_All_npt[i].im);

            lrvir[XX][XX] += tmp * (1 - 2 * k1 * k1 / k_square) + tmp_npt * k1_npt * k1_npt / (2.0*alpha);
            lrvir[XX][YY] += tmp * (-2 * k1 * k2 / k_square) + tmp_npt * k1_npt * k2_npt / (2.0 * alpha);
            lrvir[XX][ZZ] += tmp * (-2 * k1 * k3 / k_square) + tmp_npt * k1_npt * k3_npt / (2.0 * alpha);
            lrvir[YY][YY] += tmp * (1 - 2 * k2 * k2 / k_square) + tmp_npt * k2_npt * k2_npt / (2.0 * alpha);
            lrvir[YY][ZZ] += tmp * (-2 * k2 * k3 / k_square) + tmp_npt * k2_npt * k3_npt / (2.0 * alpha);
            lrvir[ZZ][ZZ] += tmp * (1 - 2 * k3 * k3 / k_square) + tmp_npt * k3_npt * k3_npt / (2.0 * alpha);            
            */

            /*
            real k1 = KKKx[i], k2 = KKKy[i], k3 = KKKz[i];
            real k_square = KKSquare[i];
            real ak = 1.0 / k_square;
            real akv = 2.0 * (1.0 / k_square - factor);
            energy_AB[q] -= 0.5 * ak * (et->Comm_All[i].re * et->Comm_All[i].re + et->Comm_All[i].im * et->Comm_All[i].im);
            real tmp = scaleVirial * scale * (1.0 / k_square) * memoryfactor[i] * (et->Comm_All[i].re * et->Comm_All[i].re + et->Comm_All[i].im * et->Comm_All[i].im); 
            lrvir[XX][XX] += tmp * (1 - akv * k1 * k1);
            lrvir[XX][YY] += tmp * (-akv * k1 * k2);
            lrvir[XX][ZZ] += tmp * (-akv * k1 * k3);
            lrvir[YY][YY] += tmp * (1 - akv * k2 * k2);
            lrvir[YY][ZZ] += tmp * (-akv * k2 * k3);
            lrvir[ZZ][ZZ] += tmp * (1 - akv * k3 * k3);
            */
            //if ((et->Step%63==0)&&(et->me == 0) && (i==0))
             //   std::cout << "S = " << et->S << "   S_memory = " << et->S_memory << "    memoryfactor = " << memoryfactor << std::endl;
        }

        //real virialZ = 0.0, virialX = 0.00, virialY = 0.00;
        for (int k = -et->kz_max; k <= et->kz_max; k++)
        {
            if (k != 0)
            {
                real k3 = k_rhok[k+et->kz_max];
                real k_square = ksqrt[k + et->kz_max];
                real ak = std::exp(k_square * factor) / k_square;
                real akv = 2.0 * ak * (1.0 / k_square - factor);
                
                real rhok2 = (rhok[et->kz_max + k][0] * rhok[et->kz_max + k][0] + rhok[et->kz_max + k][1] * rhok[et->kz_max + k][1]);
                
                lrvir[ZZ][ZZ] += scaleNew * (ak * rhok2 - scale * akv * rhok2 * k3 * k3);
                lrvir[YY][YY] += scaleNew * (ak * rhok2);
                lrvir[XX][XX] += scaleNew * (ak * rhok2);

                //virialX += scaleNew * (ak * rhok2);
                //virialY += scaleNew * (ak * rhok2);
                //virialZ += scaleNew * (ak * rhok2 - k3 * k3 * scale * akv * rhok2);

                addenergy += rhok2 / k_square * std::exp(k_square * factor);
            }
        }

        //printf("The additional virial is %lf %lf %lf The et->size is %d\n",virialX,virialY,virialZ, et->size);

        //printf("The additional virial is %lf %lf %lf The et->size is %d\n", virialX, virialY, virialZ, et->size);
        //printf("The additional virial is %lf %lf %lf The et->size is %d\n", lrvir[XX][XX], lrvir[YY][YY], lrvir[ZZ][ZZ], et->size);  
        //printf("Virial is %lf %lf %lf\n", lrvir[XX][YY], lrvir[YY][ZZ], lrvir[XX][ZZ]);


         /*
        for (int k = -et->kz_max; k <= et->kz_max; k++)
        {
            if (k != 0)
            {
                addenergy += (rhok[et->kz_max + k][0] * rhok[et->kz_max + k][0] + rhok[et->kz_max + k][1] * rhok[et->kz_max + k][1]) / (k_rhok[k + et->kz_max] * k_rhok[k + et->kz_max]) * std::exp(-k_rhok[k + et->kz_max] * k_rhok[k + et->kz_max] / (4 * alpha));
            }
        } 
        */
        //t1 = MPI_Wtime();
        //if (et->me == 0)std::cout << "Time of the second part is " << (t1 - t5) * 1000 << "  ms" <<"   Factor is "<<memoryfactor[1]<< std::endl;
    }

    if ( et->me == 0) {
        //std::cout << "S = " << et->S << "   S_memory = " << et->S_memory << "    memoryfactor = " << memoryfactor[0] << std::endl;
    }

    if (!bFreeEnergy)
    {
        energy = energy_AB[0];
    }
    else
    {
        energy = (1.0 - lambda) * energy_AB[0] + lambda * energy_AB[1];
        *dvdlambda += scaleRecip * (energy_AB[1] - energy_AB[0]);
    }

   // if (et->me == 0)
    //    std::cout << lrvir[0][0] << "     " << lrvir[1][1] << "    " << lrvir[2][2] << std::endl;

    lrvir[YY][XX] = lrvir[XX][YY];
    lrvir[ZZ][XX] = lrvir[XX][ZZ]; 
    lrvir[ZZ][YY] = lrvir[YY][ZZ];

    energy *= (-4 * M_PI * (et->S / (et->P + 0.00)) * (gmx::c_one4PiEps0 / epsilonR) / (V + 0.00));
    
    //Kz部分的能量
    energy += 2 * M_PI / V * addenergy * (gmx::c_one4PiEps0 / epsilonR);

    //printf("The additional energy is %lf\n", 2 * M_PI / V * addenergy * (gmx::c_one4PiEps0 / epsilonR));
     

    //是否存在这个全局归约？好像还真存在
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            lrvir[i][j] = lrvir[i][j] / (et->size + 0.00);
        }
    }
    energy = energy / (et->size + 0.00);

    //printf("The energy is %lf\n", energy);
    //printf("The additional virial is %lf %lf %lf The et->size is %d\n", lrvir[XX][XX], lrvir[YY][YY], lrvir[ZZ][ZZ], et->size);

    if((et->Step%1000)==0&&et->me==0)
        std::cout << boxDiag[0] << "     " << boxDiag[1] << "    " << boxDiag[2] << "     " << V << std::endl;
    return energy;
}

real ewald_charge_correction(const t_commrec*            commrec,
                             const real                  epsilonR,
                             const real                  ewaldcoeffQ,
                             gmx::ArrayRef<const double> qsum,
                             const real                  lambda,
                             const matrix                box,
                             real*                       dvdlambda,
                             tensor                      vir)

{
    real enercorr = 0;

    if (MASTER(commrec))
    {
        /* Apply charge correction */
        real vol = box[XX][XX] * box[YY][YY] * box[ZZ][ZZ];

        real fac = M_PI * gmx::c_one4PiEps0 / (epsilonR * 2.0 * vol * vol * gmx::square(ewaldcoeffQ));

        real qs2A = qsum[0] * qsum[0];
        real qs2B = qsum[1] * qsum[1];

        real vc = (qs2A * (1 - lambda) + qs2B * lambda) * fac;

        enercorr = -vol * vc;

        *dvdlambda += -vol * (qs2B - qs2A) * fac;

        for (int d = 0; d < DIM; d++)
        {
            vir[d][d] += vc;
        }
    }

    return enercorr;
}
