#define HASH_SIZE 128
#define PARALLEL

#include <fht.h>
#include <iostream>

using Image = std::vector<std::vector<DataType>>;

int main(int argc, char* argv[])
{
    // Define global 3D array sizes
    size_t cols = pow(2, 9);
    size_t image_num = 100;

    if (argc > 3) {
        printf("Only R (resolution) and N (number of images)\n");
        return 1;
    } else if (argc == 3) {
        sscanf(argv[1], "%d", &cols);
        sscanf(argv[2], "%d", &image_num);
    } else if (argc == 2) {
        printf("Standart image num = 100\n");
        return 1;
    }
    const size_t rows = cols;

    // input data
    /* 1D:
    std::vector<DataType> a1(cols);
    for (size_t j = 0; j < cols; ++j) {
        a1[j] = static_cast<DataType>(20 * cols + j + 2) / cols;
    }
    */

    /* 2D:
    std::vector<std::vector<DataType>> a2(cols);
    for (size_t j1 = 0; j1 < cols; ++j1) {
        a2[j1].resize(rows);
        for (size_t j2 = 0; j2 < rows; ++j2)
            a2[j1][j2] = static_cast<DataType>(cols + j1 + j2 + 2) / cols;
    }
    */
    /* 3D: */
    std::vector<Image> a3(image_num);
    for (size_t j1 = 0; j1 < image_num; ++j1) {
        a3[j1].resize(cols);
        for (size_t j2 = 0; j2 < cols; ++j2) {
            a3[j1][j2].resize(rows);
            for (size_t j3 = 0; j3 < cols; ++j3) {
                a3[j1][j2][j3] = static_cast<DataType>(rows + j1 + j2 + j3 + 2) / (cols * image_num);
            }
        }
    }

    double common_start, common_finish;
    common_start = clock() / static_cast<double>(CLOCKS_PER_SEC);

    //std::vector<DataType> kernel;
    //initializeKernelHost(&kernel, a2[0].size());
    //std::vector<DataType> test;

    for (int i0 = 0; i0 < image_num; ++i0) {
        for (int direction = 0; direction < 1; ++direction) {
#ifdef PARALLEL
#pragma omp parallel for
#endif
            for (int i = 0; i < cols; ++i) {
                fht1d(&a3[i0][i]);
                //test = dht1d(a2[i], kernel);
            }
        }
    }

    common_finish = clock() / static_cast<double>(CLOCKS_PER_SEC);
    showTime(common_start, common_finish, "Common time");
    // writeData(a1, std::ios_base::out, "input");
    // writeData(test, std::ios_base::app, "output");

    return 0;
}
