/*******************************************************************************
Vendor: Xilinx
Associated Filename: krnl_vadd.cpp
Purpose: Vitis vector addition example
*******************************************************************************
Copyright (C) 2019 XILINX, Inc.

This file contains confidential and proprietary information of Xilinx, Inc. and
is protected under U.S. and international copyright and other intellectual
property laws.

DISCLAIMER
This disclaimer is not a license and does not grant any rights to the materials
distributed herewith. Except as otherwise provided in a valid license issued to
you by Xilinx, and to the maximum extent permitted by applicable law:
(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX
HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
in contract or tort, including negligence, or under any other theory of
liability) for any loss or damage of any kind or nature related to, arising under
or in connection with these materials, including for any direct, or any indirect,
special, incidental, or consequential loss or damage (including loss of data,
profits, goodwill, or any type of loss or damage suffered as a result of any
action brought by a third party) even if such damage or loss was reasonably
foreseeable or Xilinx had been advised of the possibility of the same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any
application requiring fail-safe performance, such as life-support or safety
devices or systems, Class III medical devices, nuclear facilities, applications
related to the deployment of airbags, or any other applications that could lead
to death, personal injury, or severe property or environmental damage
(individually and collectively, "Critical Applications"). Customer assumes the
sole risk and liability of any use of Xilinx products in Critical Applications,
subject only to applicable laws and regulations governing limitations on product
liability.

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
ALL TIMES.

*******************************************************************************/

//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Demonstrate Vector Add Kernel
//

//#define BUFFER_SIZE 256
//#define DATA_SIZE 4096 
////TRIPCOUNT identifier
//const unsigned int c_len = DATA_SIZE / BUFFER_SIZE;
//const unsigned int c_size = BUFFER_SIZE;

/*
    Vector Addition Kernel Implementation 
    Arguments:
        in1   (input)     --> Input Vector1
        in2   (input)     --> Input Vector2
        out_r   (output)    --> Output Vector
        size  (input)     --> Size of Vector in Integer

    extern "C" {
    void krnl_vadd(const unsigned int *in1, // Read-Only Vector 1
          const unsigned int *in2, // Read-Only Vector 2
          unsigned int *out_r,     // Output Result
          int size                 // Size in integer
    ) {

    unsigned int v1_buffer[BUFFER_SIZE];   // Local memory to store vector1

    //Per iteration of this loop perform BUFFER_SIZE vector addition
    for (int i = 0; i < size; i += BUFFER_SIZE) {
       #pragma HLS LOOP_TRIPCOUNT min=c_len max=c_len
        int chunk_size = BUFFER_SIZE;
        //boundary checks
        if ((i + BUFFER_SIZE) > size)
            chunk_size = size - i;

        read1: for (int j = 0; j < chunk_size; j++) {
           #pragma HLS LOOP_TRIPCOUNT min=c_size max=c_size
            v1_buffer[j] = in1[i + j];
        }

        //Burst reading B and calculating C and Burst writing 
        // to  Global memory
        vadd_writeC: for (int j = 0; j < chunk_size; j++) {
           #pragma HLS LOOP_TRIPCOUNT min=c_size max=c_size
            //perform vector addition
            out_r[i+j] = v1_buffer[j] + in2[i+j];
        }

    }
    }
    }
*/
/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

/*******************************************************************************

Vitis Key Concept :

    This is a matrix multiplication example which showcases the "Systolic Array"
    based algorithm design. Systolic array type of implementation is well suited
    for FPGAs.

*******************************************************************************/

/*

Kernel Description :

    This kernel is a systolic array based matrix multiplication. Though the
    maximum size of the input matrices are restricted to a smaller MAX_SIZE, it
    is still possible to use this approach and get better performance for larger
    matrices by using tiling.

    Arguments :

        int *a     (input )  --> Input  Matrix A
        int *b     (input )  --> Input  Matrix B
        int *c     (output)  --> Output Matrix
        int  a_row (input )  --> Row Size Matrix A
        int  a_col (input )  --> Col Size Matrix A
        int  b_col (input )  --> Col Size Matrix B

    Kernel Configuration :

        Max Size    --> 16

    Note :
        Max Size is dependent on the available DSP resources in the FPGA
*/

#include <stdio.h>
#include <hls_stream.h>

// Maximum Array Size
#define MAX_SIZE 8
#define SHIFT_BUFFER_LEN (MAX_SIZE * 2 -1)
#define DOT_PROD_SIZE 1

// TRIPCOUNT identifier
const unsigned int c_size = MAX_SIZE;

extern "C" {
void krnl_mmult(const int* a, // Read-Only Matrix A
           const int* b, // Read-Only Matrix B
           int* c,       // Output Result
           int a_row,    // Matrix A Row Size
           int a_col,    // Matrix A Col Size
           int b_col     // Matrix B Col Size
           ) {

    int b_row = a_col;
    int c_row = a_row;
    int c_col = b_col;

    // Local memory to store input and output matrices
    // int burstA[MAX_SIZE][MAX_SIZE];
    // #pragma HLS ARRAY_PARTITION variable = burstA dim = 2 complete

    static hls::stream<int> c_stream("localC_stream");

    int localA[MAX_SIZE][SHIFT_BUFFER_LEN];
    #pragma HLS ARRAY_PARTITION variable = localA dim = 1 complete
    
    int localB[SHIFT_BUFFER_LEN][MAX_SIZE];
    #pragma HLS ARRAY_PARTITION variable = localB dim = 2 complete

    int localC[MAX_SIZE][MAX_SIZE];
    #pragma HLS ARRAY_PARTITION variable = localC dim = 0 complete

    int bufA[MAX_SIZE][MAX_SIZE];
    #pragma HLS ARRAY_PARTITION variable = bufA dim = 0 complete

    int bufB[MAX_SIZE][MAX_SIZE];
    #pragma HLS ARRAY_PARTITION variable = bufB dim = 0 complete    

    int bufC[MAX_SIZE * MAX_SIZE + 1];
    #pragma HLS ARRAY_PARTITION variable = bufC dim = 0 complete

    // Burst reads on input matrices from global memory
    // Read Input A
    // Auto-pipeline is going to apply pipeline to these loops
    
    // BurstA:
    // for (int i = 0; i < MAX_SIZE; i ++ ){
    //     BurstA_inner:
    //     for (int j = 0; j < MAX_SIZE; j ++){
    //         burstA[i][j] = a[i*MAX_SIZE + j];
    //     }
    // }

    int local_A_j;
    readA:
    for (int i = 0; i < MAX_SIZE; i++) {
        readA_inner:
        for (int j = 0; j < MAX_SIZE; j ++){
            //shift A to fit systolic array access pattern
            local_A_j = SHIFT_BUFFER_LEN - MAX_SIZE - i + j;
            localA[i][local_A_j] = a[i * MAX_SIZE + j];
        }    
    }


    fillA:
    for (int i = 0; i < MAX_SIZE; i ++) {
        #pragma HLS UNROLL
        fillA_inner:
        for (int j = 0; j < SHIFT_BUFFER_LEN; j++){
            #pragma HLS UNROLL
            //fill the rest with 0
            if ((j < SHIFT_BUFFER_LEN - MAX_SIZE - i) || (j >= SHIFT_BUFFER_LEN - i)){
                localA[i][j] = 0;
            }
        }    
    }

    // Read Input B
    
    int local_B_i;
    readB:
    for (int i = 0; i < MAX_SIZE; i ++) {
        readB_inner:
        for (int j = 0; j < MAX_SIZE; j++){
            //shift B to fit systolic array access pattern
            local_B_i = SHIFT_BUFFER_LEN - MAX_SIZE - j + i;
            localB[local_B_i][j] = b[i * MAX_SIZE + j];
        }    
    }


    fillB:
    for (int j = 0; j < MAX_SIZE; j ++) {
        #pragma HLS UNROLL
        fillB_inner:
        for (int i = 0; i < SHIFT_BUFFER_LEN; i++){
            #pragma HLS UNROLL
            //fill the rest with 0
            if ((i < SHIFT_BUFFER_LEN - MAX_SIZE - j) || (i >= SHIFT_BUFFER_LEN - j)){
                localB[i][j] = 0;
            }
        }    
    }


    fillC:
    for (int i = 0; i < MAX_SIZE; i++){
        #pragma HLS UNROLL
        fillC_inner:
        for (int j = 0; j < MAX_SIZE; j ++){
            #pragma HLS UNROLL
            bufA[i][j] = 0;
            bufB[i][j] = 0;
            localC[i][j] = 0;
        }
    }


    
    systolic1:
    for (int k = 3 * MAX_SIZE - 2; k >= 0; k--) {
    #pragma HLS pipeline
        systolic2:
        for (int i = MAX_SIZE-1; i >= 0; i--) {
            systolic3:
            for (int j = MAX_SIZE-1; j >= 0; j--) {
                int inputA = (k >= MAX_SIZE) ? localA[i][k - MAX_SIZE] : 0;
                int inputB = (k >= MAX_SIZE) ? localB[k - MAX_SIZE][j] : 0;
                bufA[i][j] = (j > 0) ? bufA[i][j-1] : inputA;
                bufB[i][j] = (i > 0) ? bufB[i-1][j] : inputB;
                localC[i][j] += bufA[i][j] * bufB[i][j];
                
                // if ((k+i+j == MAX_SIZE + 2)){
                //     c_stream << localC[i][j];
                //     c[i * MAX_SIZE + j] = c_stream.read();
                // }
            }
        }
    }
    
    


    // Burst write from output matrices to global memory
    // Burst write from matrix C
    // writeC:
    for (int i = 0; i < MAX_SIZE; i ++ ){
        writeC_inner:
        for (int j = 0; j < MAX_SIZE; j ++){
            c[i * MAX_SIZE + j] = localC[i][j];
        }
    }


/*
    int b_row = a_col;
    int c_row = a_row;
    int c_col = b_col;

    // Local memory to store input and output matrices
    int localA[MAX_SIZE][MAX_SIZE];
    #pragma HLS ARRAY_PARTITION variable = localA dim = 1 complete

    int localB[MAX_SIZE][MAX_SIZE];
    #pragma HLS ARRAY_PARTITION variable = localB dim = 2 complete

    int localC[MAX_SIZE][MAX_SIZE];
    #pragma HLS ARRAY_PARTITION variable = localC dim = 0 complete

    // Burst reads on input matrices from global memory
    // Read Input A
    // Auto-pipeline is going to apply pipeline to these loops
    readA:
    for (int loc = 0, i = 0, j = 0; loc < a_row * a_col; loc++, j++) {
    #pragma HLS LOOP_TRIPCOUNT min = c_size* c_size max = c_size * c_size
        if (j == a_col) {
            i++;
            j = 0;
        }
        localA[i][j] = a[loc];
    }

    // Read Input B
    readB:
    for (int loc = 0, i = 0, j = 0; loc < b_row * b_col; loc++, j++) {
    #pragma HLS LOOP_TRIPCOUNT min = c_size* c_size max = c_size * c_size
        if (j == b_col) {
            i++;
            j = 0;
        }
        localB[i][j] = b[loc];
    }

// Perform systolic matrix multiply
// local matrices localA and localB have been partitioned in dimensions
// 1 and 2 respectively. local matrix C has been partitioned completely

// This partitioning enables to access MAX_SIZE elements in parallel in
// the local matrices. Because of the mode of access of array elements,
// we are able to perform MAX_SIZE*MAX_SIZE operations in parallel.

// Note : i, j and k loops are interchanged.

// The top loop systolic1 runs only for a_col iterations instead of
// MAX_SIZE like the inner loops. The inner loops have fixed loop
// iteration counts to enable complete unroll

// The following diagram explains how the matrix multiply happens
//
//        B_0        B_1        B_2        B_3
//         |          |          |          |
//         v          v          v          v
//        ___        ___        ___        ___
//       |   |      |   |      |   |      |   |
//  A0_->|C00| ---- |C01| ---- |C02| ---- |C03|
//       |___|      |___|      |___|      |___|
//         |          |          |          |
//        ___        ___        ___        ___
//       |   |      |   |      |   |      |   |
//  A1_->|C10| ---- |C11| ---- |C12| ---- |C13|
//       |___|      |___|      |___|      |___|
//         |          |          |          |
//        ___        ___        ___        ___
//       |   |      |   |      |   |      |   |
//  A2_->|C20| ---- |C21| ---- |C21| ---- |C21|
//       |___|      |___|      |___|      |___|
//         |          |          |          |
//        ___        ___        ___        ___
//       |   |      |   |      |   |      |   |
//  A3_->|C30| ---- |C31| ---- |C32| ---- |C33|
//       |___|      |___|      |___|      |___|

    systolic1:
    for (int k = 0; k < a_col; k++) {
    #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
    systolic2:
        for (int i = 0; i < MAX_SIZE; i++) {
        #pragma HLS UNROLL
        systolic3:
            for (int j = 0; j < MAX_SIZE; j++) {
            #pragma HLS UNROLL
                // Get previous sum
                int last = (k == 0) ? 0 : localC[i][j];

                // Update current sum
                // Handle boundary conditions
                int a_val = (i < a_row && k < a_col) ? localA[i][k] : 0;
                int b_val = (k < b_row && j < b_col) ? localB[k][j] : 0;
                int result = last + a_val * b_val;

                // Write back results
                localC[i][j] = result;
            }
        }
    }

// Burst write from output matrices to global memory
// Burst write from matrix C
    writeC:
    for (int loc = 0, i = 0, j = 0; loc < c_row * c_col; loc++, j++) {
    #pragma HLS LOOP_TRIPCOUNT min = c_size* c_size max = c_size * c_size
        if (j == c_col) {
            i++;
            j = 0;
        }
        c[loc] = localC[i][j];
    }

*/  

}
}