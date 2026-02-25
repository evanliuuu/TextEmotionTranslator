
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <time.h>

/*RTX 3080 Compute Capability: 8.6
* Max grids per device: 128
* Num threads per block: 1024
* Warp size: 32
*/


/*Program Plan
* This program will read a file containing tweets and their associated joy value, tokenize each word based on the associated sentence's joy value,
* and then use the data to train a simple text processing neural network to predict the joy value of a tweet based on its words.
* The framework of the neural network will be constructed and trained with the tweet-train.txt file, 
* while the tweet-test.txt file will be used to test the accuracy of the neural network.
* Each word and its associated token will be processed and assigned a default weight, which will be used to calculate the joy value of the tweet.
* Then, the neural network will be trained to adjust these weights based on the joy value of the tweet.
* The weight adjustment will be done using the back propagation algorithm, which will be implemented in CUDA to speed up the process.
* The algorithm will utilize the error between the predicted joy value and the actual joy value to adjust the weights of the words in the tweet.
*   This adjustment will be done by calculating the gradient of the error with respect to each weight, and then updating the weights in the opposite direction of the gradient.
* The overall joy value of the tweet will be calculated by summing the weighted contributions, averaged against the number of words in the tweet.
* The program will also include a histogram of the words in the tweets, which will be used to visualize the distribution of words and their associated joy values.
* 
* Math:
*   Average Joy Value = (Sum of Weighted Contributions) / (Number of Words in Tweet)
*   Weighted Contribution = All words in data averaged against the joy value of the tweet
*/


cudaError_t forwardPassWithCuda(float* bins_h, unsigned int* tokenMap_h, float* joyMap_h, unsigned int size, unsigned int binSize);
cudaError_t backPropagation(float* error, float* tokenWeights, unsigned int* wordsPerSentence, unsigned int* tokenMapVector, std::unordered_map<std::string, unsigned int> wordMap, std::vector<std::vector<std::string>> sentences, std::vector<float> joyData, const unsigned int numSentences, const unsigned int numTraining, const float learningRate);
std::vector<std::vector<std::string>> parseTweets(const std::string& filename);
std::vector<float> parseJoy(const std::string& filename);
void tokenizeWords(const std::vector<std::vector<std::string>>& sentences, std::unordered_map<std::string, unsigned int>& wordMap, std::vector<unsigned int>& tokenMap, const std::vector<float> joyData, std::vector<float>& joyMap);
void initVector(unsigned int** tokenMapVector, float** joyMapVector, std::vector<unsigned int> tokenMap, std::vector<float> joyMap);
void errorCalculate(float** error, float* tokenWeights, unsigned int** wordsPerSentence, std::unordered_map<std::string, unsigned int> wordMap, std::vector<std::vector<std::string>> sentences, std::vector<float> joyData, unsigned int numSentences, unsigned int numTrainedTokens, float averageTokenWeight);
std::string cleanWord(const std::string& word);
void calcTokenAvgErr(float* tokenAvgErr, float* error, unsigned int* tokenMapVector, unsigned int* wordsPerSentence, unsigned int numSentences, unsigned int numTokens);
void validateResults(std::unordered_map<std::string, unsigned int> wordMap, const std::vector<std::vector<std::string>>& sentences, const std::vector<float>& joyData, float* tokenWeights, const unsigned int* tokenMapVector, unsigned int numSentences, unsigned int numTokens, unsigned int numTrainedTokens);
void calculateJoy(const std::string& inputSentence, float* tokenWeights, const std::unordered_map<std::string, unsigned int>& wordMap);

__global__ void averageJoyKernel(
    float* bins, 
    float* joySums,
    unsigned int* tokenCounts,
    unsigned int numBins)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numBins) {
        bins[i] = joySums[i] / (tokenCounts[i] > 0 ? tokenCounts[i] : 1); // Avoid division by zero
    }
}

__global__ void aggregrateTokenKernel(
    const unsigned int* tokenMap,
    unsigned int* global_token_counts,
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < size; i += stride) {
        atomicAdd(&global_token_counts[tokenMap[i]], 1);
    }
}

__global__ void aggregateJoyKernel(
    const float* joyMap,
    const unsigned int* tokenMap,
    float* global_joy_sums,
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; i < size; i += stride) {
        atomicAdd(&global_joy_sums[tokenMap[i]], joyMap[i]);
    }
}

__global__ void adjustWeightsKernel(
    float* tokenWeights, 
    float* tokenAvgError, 
    unsigned int numTokens, 
    float learningRate)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numTokens) {
        // Adjust the weight of the token based on the average error and learning rate
        tokenWeights[i] -= learningRate * tokenAvgError[i];
        // Ensure weights do not go below zero
        if (tokenWeights[i] < 0.0f) {
            tokenWeights[i] = 0.0f;
        }else if (tokenWeights[i] > 1.0f) {
            tokenWeights[i] = 1.0f; // Cap the weights at 1.0
		}
    }
}

/* Remaining Tasks:
* 1) Back propagation algorithm to adjust the weights of the words in the tweet based on the error between the predicted joy value and the actual joy value.
* 2) Validate with tweet-test.txt file to test the accuracy of the neural network.
* 3) Implement an interface to type in a sentence and get the predicted joy value.
*/

// Back prop breakdown:
// Take the average error of the sentences each word appears in, AKA have all associated sentences for each word's average error happy = .51, happy = 0.49 10sentenceavgerr = .18, .21
// Use the average error data to adjust the weights of each word to minimize average error per sentence
//    Carrot and Stick approach
//    Brute force increasing and decreasing weight of each word by learningRate (0.01) and checking the average error of the sentences each word appears in
//    If average error increases, terminate branch, if average error decreases, continue branch
// Run new weight data through errorCheck to get new error values for each sentence, then repeat the process until the error converges to a minimum value





int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max shared memory per block: %zu\n", prop.sharedMemPerBlock);
	clock_t timeStart = clock();
    std::vector<std::vector<std::string>> tweets = parseTweets("tweet-train.txt");
	std::vector<float> joyData = parseJoy("tweet-train.txt");

	timeStart = clock() - timeStart;
	printf("Time taken to parse the file: %f seconds\n", static_cast<double>(timeStart) / CLOCKS_PER_SEC);

    std::unordered_map<std::string, unsigned int> wordMap;
	std::vector<unsigned int> tokenMap;
    std::vector<float> joyMap;
	tokenizeWords(tweets, wordMap, tokenMap, joyData, joyMap);

	//word map is a map of words to their unique token ID
	//tokenMap is a translation of the words in the sentences to their token ID
	//joyMap is a vector of the joy values for each tokenized word, corresponding to the sentence's joy value
	unsigned int* tokenMapVector;
	float* joyMapVector;
	initVector(&tokenMapVector, &joyMapVector, tokenMap, joyMap);
    if (0) {
        for (int i = 0; i < tokenMap.size(); i++) {
            printf("Token: %u, Joy Value: %f\n", tokenMapVector[i], joyMapVector[i]);
        }
    }
    
    unsigned int numBins = wordMap.size();
    float* tokenWeights = (float*)malloc(numBins * sizeof(float));
    cudaError_t cudaStatus = forwardPassWithCuda(tokenWeights, tokenMapVector, joyMapVector, tokenMap.size(), numBins);
    
    float* error;
	unsigned int* wordsPerSentence;
	errorCalculate(&error, tokenWeights, &wordsPerSentence, wordMap, tweets, joyData, tweets.size(), numBins, 0);

    backPropagation(error, tokenWeights, wordsPerSentence, tokenMapVector, wordMap, tweets, joyData, tweets.size(), 193, 0.1);

    timeStart = clock();
	std::vector<std::vector<std::string>> tweetsTest = parseTweets("tweet-test.txt");
	std::vector<float> joyDataTest = parseJoy("tweet-test.txt");
    timeStart = clock() - timeStart;
    printf("Time taken to parse the file: %f seconds\n", static_cast<double>(timeStart) / CLOCKS_PER_SEC);

	// Tokenize the test tweets
	std::vector<unsigned int> tokenMapTest;
	std::vector<float> joyMapTest;
	tokenizeWords(tweetsTest, wordMap, tokenMapTest, joyDataTest, joyMapTest); //Tokenize the test tweets using the same wordMap
	
    // Initialize vectors for the test tweets
	unsigned int* tokenMapVectorTest;
	float* joyMapVectorTest;
	initVector(&tokenMapVectorTest, &joyMapVectorTest, tokenMapTest, joyMapTest);

	//Validate the neural network with the test tweets
	validateResults(wordMap, tweetsTest, joyDataTest, tokenWeights, tokenMapVectorTest, tweetsTest.size(), wordMap.size(), numBins);

    //Default Token Weight = 0.581
	//Read user input for a sentence and calculate the joy value
	std::string inputSentence;
	std::cout << "Enter a sentence to calculate its joy value (type 'exit' to quit):" << std::endl;
    while (std::getline(std::cin, inputSentence)) {
        if (inputSentence == "exit") {
            break; // Exit the loop if the user types 'exit'
		}
        calculateJoy(inputSentence, tokenWeights, wordMap);
    }

    //display word map contents and token weight values
    if (0) {
        int i = 0;
        for (const auto& pair : wordMap) {
            printf("Word: %s, Joy Weight: %f, Token ID: %u\n", pair.first.c_str(), tokenWeights[i], pair.second);
            i++;
        }
    }
    //display new wordmap contents
    if (0) {
        for (int i = wordMap.size() - numBins; i < wordMap.size(); i++) {
            auto it = std::next(wordMap.begin(), i);
            printf("New Word: %s, Token ID: %u\n", it->first.c_str(), it->second);
		}
    }
    //display contents of the tokenMap and joyMap vectors
    if (0) {
        for (unsigned int i = 0; i < tokenMap.size(); i++) {
            std::cout << "Token: " << tokenMapVector[i]
                << ", Joy Value: " << joyMapVector[i] << std::endl;
        }
	}
    // print out average joy value (0.678) for the tokens
    if (0) {
		float averageJoyValue = 0.0f;
        for (unsigned int i = 0; i < numBins; i++) {
			averageJoyValue += tokenWeights[i];
        }
		averageJoyValue /= numBins;
		printf("Average Joy Value: %f\n", averageJoyValue);
    }
    // display size of vectors
    if (1) {
        // Test print the parsed data
        if (0) {
            for (const auto& row : tweets) {
                for (const auto& value : row) {
                    std::cout << value << "\t";
                }
                std::cout << std::endl;
            }
            for (const auto& joy : joyData) {
                std::cout << joy << std::endl;
            }

        }
        std::cout << "Parsed " << joyData.size() << " joy values from the file." << std::endl;
        std::cout << "Parsed " << tweets.size() << " rows from the file." << std::endl;
		std::cout << "Parsed " << wordMap.size() << " unique words from the files." << std::endl;
		std::cout << "Parsed " << tokenMap.size() << " tokens from the file." << std::endl;
		std::cout << "Parsed " << joyMap.size() << " joy values from the tokenized words." << std::endl;
		std::cout << "Parsed " << tokenMapTest.size() << " tokens from the test tweets." << std::endl;
		std::cout << "Parsed " << joyMapTest.size() << " joy values from the test tweets." << std::endl;
        std::cout << "Number of new tokens: " << wordMap.size() - numBins << std::endl;
    }


    // Free allocated memory
    free(tokenWeights);
    free(error);
    free(wordsPerSentence);
    free(tokenMapVector);
    free(joyMapVector);
    // Free the wordMap
    for (auto& pair : wordMap) {
        pair.second = 0; // Clear the values in the map
    }
    wordMap.clear();
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
	}
    return 0;
}

//Necessary data:
// error (float*) - the error for each sentence
// bins, AKA tokenWeights (float*) - the weights for each word in the tweet, indexed by the token ID
// wordsPerSentence (unsigned int*) - the number of words in each sentence, indexed by the sentence index
// tokenMapVector (unsigned int*) - each word in the tweet translated to its token ID
// wordMap (unordered_map<string, unsigned int>) - a map of words to their unique token ID

/* 
* Program Flow:
* error (vector of float errors for each sentence)
* Use error, wordsPerSentence and tokenMapVector to make tokenAvgError
* tokenAvgError (average error for all sentences associated with each token) (c++ function)
* 
* Training Loop:
* 1) adjustWeightsKernel (Adjust weights of each token based on learning rate and tokenAvgError)
* 2) errorCalculate (Calculate new error for each sentence)
* 3) tokenAvgError (Calculate average error for all sentences associated with each token)
*/

cudaError_t backPropagation(
    float* error, 
    float* tokenWeights_h, 
    unsigned int* wordsPerSentence, 
    unsigned int* tokenMapVector, 
    std::unordered_map<std::string, unsigned int> wordMap,
	std::vector<std::vector<std::string>> sentences,
	std::vector<float> joyData,
    const unsigned int numSentences,
    const unsigned int numTraining,
    const float learningRate) 
{
    float* tokenAvgError_h = (float*)malloc(wordMap.size() * sizeof(float));
	calcTokenAvgErr(tokenAvgError_h, error, tokenMapVector, wordsPerSentence, numSentences, wordMap.size());

	// Allocate memory, set up device, and train the neural network
    cudaError_t cudaStatus;
    float* tokenAvgError_d;
    float* tokenWeights_d;

    clock_t timeStart = clock();
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocating device variables
    cudaStatus = cudaMalloc((void**)&tokenAvgError_d, wordMap.size() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to allocate device memory for tokenAvgError!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&tokenWeights_d, wordMap.size() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to allocate device memory for tokenWeights!");
        goto Error;
    }
    cudaDeviceSynchronize();

    // Training Loop
    for (int i = 0; i < numTraining; i++) {
		printf("Progress: %d/%d, ", i + 1, numTraining);
        // Copy host variables to device
        cudaStatus = cudaMemcpy(tokenAvgError_d, tokenAvgError_h, wordMap.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Unable to copy tokenAvgError to device!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(tokenWeights_d, tokenWeights_h, wordMap.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Unable to copy tokenWeights to device!");
            goto Error;
        }
        cudaDeviceSynchronize();
        /*Launch kernel to adjust weights (num weights = wordMap.size())
		* There are ceil(wordMap.size() / threadsPerBlock) blocks needed to process the data
        */
		unsigned int threadsPerBlock = 256; // Number of threads per block
		unsigned int blocks = (wordMap.size() + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed
		//printf("Launching kernel with %u blocks and %u threads per block\n", blocks, threadsPerBlock);

		adjustWeightsKernel << <blocks, threadsPerBlock >> > (tokenWeights_d, tokenAvgError_d, wordMap.size(), learningRate);

        // Copy device variables from host
        cudaStatus = cudaMemcpy(tokenWeights_h, tokenWeights_d, wordMap.size() * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Unable to copy tokenWeights from device!");
            goto Error;
		}
		if (i < numTraining - 1) { // If this is not the last iteration, recalculate the error
            errorCalculate(&error, tokenWeights_h, &wordsPerSentence, wordMap, sentences, joyData, numSentences, wordMap.size(), 0); //error might bug
            calcTokenAvgErr(tokenAvgError_h, error, tokenMapVector, wordsPerSentence, numSentences, wordMap.size());
        }
    }
	timeStart = clock() - timeStart;

    //Test print the average error for each token
    if (0) {
        for (int i = 0; i < wordMap.size(); i++) {
            printf("Token ID: %u, Average Error: %f\n", i, tokenAvgError_h[i]);
        }
    }
	printf("Time taken to train neural network: %f seconds\n", static_cast<double>(timeStart) / CLOCKS_PER_SEC);

    
Error:
	cudaFree(tokenAvgError_d);
	cudaFree(tokenWeights_d);
	free(tokenAvgError_h);
    return cudaStatus;
}

cudaError_t forwardPassWithCuda(
    float* bins_h,
    unsigned int* tokenMap_h,
    float* joyMap_h,
    unsigned int size,
    unsigned int binSize)
{
    // This function sets up bins equal to the number of unique tokens, and then iterates through the tokenMap and joyMap vectors to calculate the average joy value for each token.
    cudaError_t cudaStatus;
    float* bins_d;
    unsigned int* tokenMap_d;
    float* joyMap_d;
    float* joySums_d;
    unsigned int* tokenCounts_d;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    clock_t timeStart = clock();

    // Allocating device variables
    cudaStatus = cudaMalloc((void**)&bins_d, binSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to allocate device memory for bins!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&tokenMap_d, size * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to allocate device memory for tokenMap!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&joyMap_d, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to allocate device memory for joyMap!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&joySums_d, binSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to allocate device memory for joySums!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&tokenCounts_d, binSize * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to allocate device memory for tokenCounts!");
        goto Error;
    }
    cudaDeviceSynchronize();

    // Copy host variables to device
    cudaStatus = cudaMemset(bins_d, 0, binSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to set device memory!");
        goto Error;
    }
    cudaStatus = cudaMemset(joySums_d, 0, binSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to set joySums device memory!");
        goto Error;
    }
    cudaStatus = cudaMemset(tokenCounts_d, 0, binSize * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to set tokenCounts device memory!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(tokenMap_d, tokenMap_h, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to copy tokenMap to device!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(joyMap_d, joyMap_h, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to copy joyMap to device!");
        goto Error;
    }
    cudaDeviceSynchronize();

    // Launch kernel to process forward pass
    // There are ceil(size / threadsPerBlock) blocks needed to process the data
    unsigned int threadsPerBlock = 256; // Number of threads per block
    unsigned int blocks = (size + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed
    printf("Launching kernel with %u blocks and %u threads per block\n", blocks, threadsPerBlock);
    printf("binSize: %u, size: %u\n", binSize, size);

    aggregrateTokenKernel << <blocks, threadsPerBlock >> > (tokenMap_d, tokenCounts_d, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "aggregrateTokenKernel launch failed: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    aggregateJoyKernel << <blocks, threadsPerBlock >> > (joyMap_d, tokenMap_d, joySums_d, size);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "aggregateJoyKernel launch failed: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    blocks = (binSize + threadsPerBlock - 1) / threadsPerBlock; // Recalculate number of blocks needed for the averageJoyKernel
    averageJoyKernel << <blocks, threadsPerBlock >> > (bins_d, joySums_d, tokenCounts_d, binSize);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "averageJoyKernel launch failed: %s\n", cudaGetErrorString(err));
        goto Error;
    }

    //Copy device variables from host
    cudaStatus = cudaMemcpy(bins_h, bins_d, binSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Unable to copy bins from device!");
        goto Error;
    }
    cudaDeviceSynchronize();

    //Verify results
    if (0) {
        for (unsigned int i = 0; i < binSize; i++) {
            printf("Bin %u: %f, \n", i, bins_h[i]);
        }
    }

    timeStart = clock() - timeStart;
    printf("Time taken to process histogram: %f seconds\n", static_cast<double>(timeStart) / CLOCKS_PER_SEC);

Error:
    cudaFree(bins_d);
    cudaFree(tokenMap_d);
    cudaFree(joyMap_d);
    cudaFree(joySums_d);
    cudaFree(tokenCounts_d);
    return cudaStatus;
}

//This function uses the initial token weights to calculate the error for each sentence based on the joy value of the sentence.
void errorCalculate(
    float** error, 
    float* tokenWeights, 
    unsigned int** wordsPerSentence,
    std::unordered_map<std::string, unsigned int> wordMap,
    std::vector<std::vector<std::string>> sentences,
    std::vector<float> joyData,
    unsigned int numSentences,
    unsigned int numTrainedTokens,
    float averageTokenWeight)
{
    //This function will use the word map to calculate the error between the token weights added into a sentence and the joy value of the sentence
    *error = (float*)malloc(numSentences * sizeof(float));
	*wordsPerSentence = (unsigned int*)malloc(numSentences * sizeof(unsigned int));
	//syntax for using error: (*error)[sentenceindex] = error value for that sentence

    unsigned int sentenceCount = 0;
    unsigned int wordCount = 0;
	float greatestError = 0.0f;
	float averageError = 0.0f;

    for (const auto& sentence : sentences) {
        // Initialize the error for the current sentence
		(*error)[sentenceCount] = 0.0f;
        for (const auto& word : sentence) {
            std::string value;
			std::string filteredWord = word; // Copy the word to a new string for filtering
			std::replace(filteredWord.begin(), filteredWord.end(), '.', ' '); // Replace periods with spaces
            std::stringstream ss(filteredWord);
            while (std::getline(ss, value, ' ')) {
                value = cleanWord(value.c_str()); // Clean the word by removing punctuation
				//This code separates each word in the sentence and replaces any punctuation with a space
				//This code will use the wordmap to find the token ID for the words in each sentence, then map the token ID to the tokenWeights vector
				//Then adding all the token weights together to get the total joy value for the sentence
				unsigned int tokenID = wordMap[value];
                if (tokenID >= numTrainedTokens) {
					(*error)[sentenceCount] += averageTokenWeight; // If the token ID is not in the trained tokens, add 0 to the error
                }
                else {
                    (*error)[sentenceCount] += tokenWeights[tokenID];
                }

                wordCount++;
            }
			float temp = (*error)[sentenceCount];
			(*error)[sentenceCount] /= wordCount; // Average the joy value for the sentence
            //printf("Sentence %u, Word Count: %u, Estimated Joy Value: %f, Actual Joy Value: %f", sentenceCount, wordCount, (*error)[sentenceCount], joyData[sentenceCount]);
			(*error)[sentenceCount] -= joyData[sentenceCount]; // Calculate the error for the sentence
			averageError += fabs((*error)[sentenceCount]); // Accumulate the absolute error for averaging
            if (fabs((*error)[sentenceCount]) > greatestError) {
                greatestError = fabs((*error)[sentenceCount]);
			}
            //printf(", Error: %f\n", (*error)[sentenceCount]);
			(*wordsPerSentence)[sentenceCount] = wordCount; // Store the number of words in the sentence
            wordCount = 0;
            sentenceCount++;
            // converts each word to their token ID and stores it in the tokenMap vector
        }
    }
	averageError /= numSentences; // Calculate the average error across all sentences
	printf("Greatest Error: %f, Average Error: %f\n", greatestError, averageError);
}

std::vector<std::vector<std::string>> parseTweets(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<std::string>> data;
    std::string line;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string value;
            int count = 0;
            while (std::getline(ss, value, '\t')) {
                if (count == 1) {
                    row.push_back(value);
                }
                count++;
            }
            data.push_back(row);
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
    return data;
}

std::vector<float> parseJoy(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<float> data;
    std::string line;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string value;
            int count = 0;
            while (std::getline(ss, value, '\t')) {
                if (count == 3) {
                    data.push_back(std::stof(value));
                }
                count++;
            }
        }
        file.close();
    }
    else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
    return data;
}

// This function will tokenize the words in the sentences and assign a unique token to each word based on its first occurrence.
void tokenizeWords(
    const std::vector<std::vector<std::string>>& sentences, 
    std::unordered_map<std::string, unsigned int>& wordMap, 
    std::vector<unsigned int>& tokenMap, 
    const std::vector<float> joyData, 
    std::vector<float>& joyMap)
{
    unsigned int tokenIndex = wordMap.size();
    unsigned int count = 0;
    unsigned int wordCount = 0;
    for (const auto& sentence : sentences) {
        for (const auto& word : sentence) {
            std::string value;
			std::string filteredWord = word; // Copy the word to a new string for filtering
			std::replace(filteredWord.begin(), filteredWord.end(), '.', ' '); // Replace commas with spaces

			std::stringstream ss(filteredWord); // basically makes the sentence a stream so we can read it word by word
            while (std::getline(ss, value, ' ')) {
                // New word, assign a new token
				value = cleanWord(value); // Clean the word by removing punctuation
                if (wordMap.find(value.c_str()) == wordMap.end()) {

                    wordMap[value.c_str()] = tokenIndex++;
                    //printf("New word found: %s, assigning token %u\n", value.c_str(), wordMap[value.c_str()]);
                }
                tokenMap.push_back(wordMap[value.c_str()]);
                joyMap.push_back(joyData[count]);
                wordCount++;
            }
            // converts each word to their token ID and stores it in the tokenMap vector
        }
        count++;
    }
}

void initVector(
    unsigned int** tokenMapVector, 
    float** joyMapVector, 
    std::vector<unsigned int> tokenMap, 
    std::vector<float> joyMap) 
{
    // Allocate memory for the vectors
    *tokenMapVector = (unsigned int*)malloc(tokenMap.size() * sizeof(unsigned int));
    *joyMapVector = (float*)malloc(joyMap.size() * sizeof(float));
    // Copy the data into the allocated memory
    std::copy(tokenMap.begin(), tokenMap.end(), *tokenMapVector);
    std::copy(joyMap.begin(), joyMap.end(), *joyMapVector);
}

std::string cleanWord(const std::string& word) {
    std::string cleaned;
    for (char c : word) {
        if (c == '\n' || c == '\r') {
            continue; // Skip newline characters
		}
        if (c <= 127 && (std::isalnum(static_cast<unsigned char>(c)) || c == '!' || c == '%' || c == '#')) { // Keep alphanumeric characters and underscores
            cleaned += c;
        }
    }
    return cleaned;
}

void calcTokenAvgErr(
    float* tokenAvgErr, 
    float* error, 
    unsigned int* tokenMapVector, 
    unsigned int* wordsPerSentence,
    unsigned int numSentences,
    unsigned int numTokens)
{
	//Calcuate the average error for each token using all sentences that the token appears in
	//Use wordsPerSentence to iterate through tokenMapVector to identify which tokens are associated with which sentences
    //Create a vector of ints to store the number of times a token appears in each sentence, indexed by the sentence
	//Additionally, add the error for each sentence to the tokenAvgErr vector, indexed by the token ID
	//Divide the total error for each token by the number of sentences the token appears in to get the average error for each token
    unsigned int currentSentence;
    unsigned int wordCount = 0;
	unsigned int* sentencesPerToken = new unsigned int[numTokens]();
    for (unsigned int i = 0; i < numTokens; i++) {
        tokenAvgErr[i] = 0.0f; // Initialize the average error for each token to 0
		sentencesPerToken[i] = 0; // Initialize the number of sentences for each token to 0
	}
    for (currentSentence = 0; currentSentence < numSentences; currentSentence++) {
        for (unsigned int i = 0; i < wordsPerSentence[currentSentence]; i++) {
			//Increment the count of sentences for the token ID inside tokenMapVector and add the sentence's error to the tokenAvgErr vector
            sentencesPerToken[tokenMapVector[wordCount]]++;
			tokenAvgErr[tokenMapVector[wordCount]] += error[currentSentence];
            wordCount++;
        }
    }

    // Calculate average error for each token
    for (unsigned int i = 0; i < numTokens; i++) {
        tokenAvgErr[i] /= sentencesPerToken[i];
	}
}

void validateResults(
	std::unordered_map<std::string, unsigned int> wordMap,
    const std::vector<std::vector<std::string>>& sentences, 
    const std::vector<float>& joyData, 
    float* tokenWeights,
    const unsigned int* tokenMapVector, 
    unsigned int numSentences, 
    unsigned int numTokens, 
    unsigned int numTrainedTokens)
{
    float* error;
	unsigned int* wordsPerSentence;
    // Now validate with the test data
    // This will use the same tokenWeights and wordMap to calculate the error for the test data
	// The results will be printed to the console
	printf("\nValidating results with tweet-test.txt...\n");
    errorCalculate(&error, tokenWeights, &wordsPerSentence, wordMap, sentences, joyData, numSentences, numTrainedTokens, 0.581);
    
    if (0) {
        for (int i = 0; i < numSentences; i++) {
            printf("Sentence %d, Estimated Joy Value: %f, Actual Joy Value: %f, Error: %f\n",
                i + 1,
                error[i] + joyData[i], // Add the error to the actual joy value to get the estimated joy value
                joyData[i],
                error[i]);
        }
    }
}

void calculateJoy(
    const std::string& inputSentence, 
    float* tokenWeights, 
    const std::unordered_map<std::string, unsigned int>& wordMap) 
{
    std::string filteredSentence = inputSentence; // Copy the word to a new string for filtering
    std::replace(filteredSentence.begin(), filteredSentence.end(), '.', ' '); // Replace commas with spaces
    std::stringstream ss(filteredSentence);
    std::string word;
    float joyValue = 0.0f;
    unsigned int wordCount = 0;
    while (std::getline(ss, word, ' ')) {
        std::string cleanedWord = cleanWord(word);
        auto it = wordMap.find(cleanedWord);
        if (it != wordMap.end()) {
            unsigned int tokenID = it->second;
            joyValue += tokenWeights[tokenID];
            wordCount++;
        }else {
            printf("Word '%s' not found in the word map. Default value 0.581 assigned.\n", cleanedWord.c_str());
			joyValue += 0.581f; // If the word is not in the wordMap, use the average token weight
            wordCount++;
        }
    }
    if (wordCount > 0) {
        joyValue /= wordCount; // Average the joy value
    } else {
        joyValue = 0.0f; // If no words were found, set joy value to 0
    }
    printf("Joy value for the input sentence (From 0.0 to 1.0): %f\n", joyValue);
}