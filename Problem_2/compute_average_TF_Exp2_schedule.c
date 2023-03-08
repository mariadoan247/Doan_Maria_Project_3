#include <stdlib.h> 
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MAX_LINE_LENGTH 1000000
#define GENE_ARRAY_SIZE 164000 
#define NUM_TETRANUCS   256
#define GENE_SIZE       10000
#define DEBUG           1


// Store gene-data here //
struct Genes {
    unsigned char* gene_sequences; // The gene-data
    int* gene_sizes;               // The gene lengths
    int  num_genes;                // The number of genes read in
}; // End Genes //




// Read Genes -------------------------- //
//      Reads in the gene-data from a file.
struct Genes read_genes(FILE* inputFile) {

    // return this
    struct Genes genes;
    genes.gene_sequences = (unsigned char*)malloc((GENE_ARRAY_SIZE * GENE_SIZE) * sizeof(unsigned char)); // array of genes
    genes.gene_sizes = (int*)malloc(GENE_ARRAY_SIZE * sizeof(int)); // array of gene lengths
    genes.num_genes = 0;                                            // number of existing genes

    // Remove the first header
    char line[MAX_LINE_LENGTH] = { 0 };
    fgets(line, MAX_LINE_LENGTH, inputFile);

    // Read in lines from the file while line exists
    int currentGeneIndex = 0;
    while (fgets(line, MAX_LINE_LENGTH, inputFile)) {

        // If the line is empty, exit loop.
        //      This if statment helps avoid errors.
        //      (strcmp returns 0 for equal strings)
        if (strcmp(line, "") == 0) {
            break;
        }

        // If line is a DNA sequence:
        //      Read in the nucleotides in order into an array
        else if (line[0] != '>') {

            int line_len = strlen(line);

            //#pragma parallel for
            for (int i = 0; i < line_len; ++i) {
                char c = line[i];
                if (c == 'A' || c == 'C' || c == 'G' || c == 'T') {
                    genes.gene_sequences[genes.num_genes * GENE_SIZE + currentGeneIndex] = c;  // put letter into gene
                    currentGeneIndex += 1;                                                     // increase currentGene size
                }
            }

        }

        // If line is a header:
        //      Reset for another gene-pass.
        else if (line[0] == '>') {
            // indicate we have another gene to read in
            genes.gene_sizes[genes.num_genes] = currentGeneIndex;
            genes.num_genes += 1;
            currentGeneIndex = 0;
        }
    }

    genes.gene_sizes[genes.num_genes] = currentGeneIndex;
    genes.num_genes += 1;


    // done
    return(genes);

} // End Read Genes //



// Process Tetranucs ------------------------ //
/* Input: A DNA sequence of length N for a gene stored at gene_index in the genes array
    Output: The TF of this gene, which is an integer array of length 256

    For each i between 0 and N-4:
            Get the windowing from i to i+3 in the DNA sequence
            This windowing is a tetranucleotide
            Convert this tetranucleotide to its array index, idx
            TF[idx]++
*/
void process_tetranucs(struct Genes genes, int* gene_TF, int gene_index) {
    // Process the current gene array
    int size = genes.gene_sizes[gene_index];        // Get size of gene
    int idx = 0;                                    // Initialize index
    int window[4];                                  // Initialize Window
    
    for (int i = 0; i <= size-4; ++i) {
        // Obtain substring tetranucleotide (A=0, C=1, G=2, T=3)
        for (int j = 0; j < 4; ++j) {
            if (genes.gene_sequences[gene_index * GENE_SIZE + i + j] == 'A') {
                window[j] = 0;
            }
            else if (genes.gene_sequences[gene_index * GENE_SIZE + i + j] == 'C') {
                window[j] = 1;
            }
            else if (genes.gene_sequences[gene_index * GENE_SIZE + i + j] == 'G') {
                window[j] = 2;
            }
            else if (genes.gene_sequences[gene_index * GENE_SIZE + i + j] == 'T') {
                window[j] = 3;
            }
        }
        // Convert tetranucleotide to its array index
        idx = window[0]*64 + window[1]*16 + window[2]*4 + window[3];
        ++gene_TF[idx];
    }

} // End Process Tetranucs //



// Main Program -------------------- //
//      Processes the tetranucleotides.
int main(int argc, char* argv[]) {
    // Check for console errors
    if (argc != 5) {
        printf("USE LIKE THIS:\ncompute_average_TF_Exp1_atomic GRCh38_latest_rna.fna average_TF.csv time.csv num_threads\n");
        exit(-1);
    }

    // Get the input file
    FILE* inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        printf("ERROR: Could not open file %s!\n", argv[1]);
        exit(-2);
    }

    // Get the output file
    FILE* outputFile = fopen(argv[2], "w");
    if (outputFile == NULL) {
        printf("ERROR: Could not open file %s!\n", argv[2]);
        fclose(inputFile);
        exit(-3);
    }

    // Get the time file
    FILE* timeFile = fopen(argv[3], "w");
    if (outputFile == NULL) {
        printf("ERROR: Could not open file %s!\n", argv[3]);
        fclose(inputFile);
        fclose(outputFile);
        exit(-4);
    }

    // Get num threads
    int thread_count = strtol(argv[4], NULL, 10);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //


    // Below is a data structure to help you access the gene-file's data:
    //      access gene like: genes.genes[gene_index*GENE_SIZE + char_index]
    //      access each gene's size like: genes.gene_sizes[gene_index]
    //      access the total number of genes like: genes.num_genes
    struct Genes genes = read_genes(inputFile);

    // Total number of tetranucs
    int* TF = (int*)calloc(NUM_TETRANUCS, sizeof(int));

    // Get the start time
    double start = omp_get_wtime();

    /*  1) Tetranuc computation
            For each gene in the list:
                Compute this gene's TF
                Add this gene's TF to the running total TF
    */ 

    // Parallelize the computations for each gene with thread_count number of
    // threads and shared variables genes and TF.
    // Fastest scheduler was dynamic.
    #pragma omp parallel for num_threads(thread_count) default(none)\
            shared(genes, TF) schedule(dynamic)
    for (int gene_index = 0; gene_index < genes.num_genes; ++gene_index) {

        // Compute this gene's TF
        int* gene_TF = (int*)calloc(NUM_TETRANUCS, sizeof(int));
        process_tetranucs(genes, gene_TF, gene_index);

        // Add this gene's TF to the running total TF
        for (int t = 0; t < NUM_TETRANUCS; ++t)
            #pragma omp atomic
            TF[t] += gene_TF[t];

        free(gene_TF);
    }


    // 2) Get the averages of each TF (as a double!) 
    double* average_TF = (double*)malloc(NUM_TETRANUCS * sizeof(double));
    // Parallelize the computations for each gene with thread_count number of
    // threads and shared variables genes, TF, and average_TF.
    // Fastest scheduler was dynamic.
    #pragma omp parallel for num_threads(thread_count) default(none)\
            shared(genes, TF, average_TF) schedule(dynamic)
    for (int t = 0; t < NUM_TETRANUCS; ++t)
        average_TF[t] = (double)TF[t] / (double)genes.num_genes;


    // Get the passed time
    double end = omp_get_wtime();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //


    // Print the average tetranucs
    for (int i = 0; i < NUM_TETRANUCS; ++i) {
        fprintf(outputFile, "%f", average_TF[i]);
        if (i < NUM_TETRANUCS - 1) fprintf(outputFile, "\n");
    }

    // Print the output time
    double time_passed = end - start;
    fprintf(timeFile, "%f", time_passed);


    // Cleanup
    fclose(timeFile);
    fclose(inputFile);
    fclose(outputFile);
    free(average_TF);
    free(TF);
    free(genes.gene_sequences);
    free(genes.gene_sizes);

    // done
    return 0;

} // End Main //
