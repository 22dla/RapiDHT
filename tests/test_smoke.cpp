#include <rapidht/transform.h>
#include <rapidht/utilities.h>

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "test_support.h" // SKIP_IF_NO_CUDA

using namespace RapiDHT;

// Element-wise closeness, reported as a single assertion.
template <typename T>
void ExpectVectorsNear(const std::vector<T>& a, const std::vector<T>& b, double tol = 1e-6)
{
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        ASSERT_NEAR(a[i], b[i], tol) << "Mismatch at index " << i;
    }
}

// ---- 1D ----
// ------- CPU ------
TEST(FDHT, Test1D_Small_CPU)
{
    size_t width = 8;
    auto mode = Modes::CPU;

    auto original = MakeData<double>({ width });
    auto transformed = original;

    HartleyTransform<double> ht(width, 0, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

TEST(FDHT, Test1D_Large_CPU)
{
    size_t width = 1 << 16; // 65536
    auto mode = Modes::CPU;

    auto original = MakeData<double>({ width });
    auto transformed = original;

    HartleyTransform<double> ht(width, 0, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

// ------- GPU ------
TEST(FDHT, Test1D_Small_GPU)
{
    SKIP_IF_NO_CUDA();
    size_t width = 8;
    auto mode = Modes::GPU;

    auto original = MakeData<double>({ width });
    auto transformed = original;

    HartleyTransform<double> ht(width, 0, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

TEST(FDHT, Test1D_Large_GPU)
{
    SKIP_IF_NO_CUDA();
    size_t width = 1 << 12; // 4096
    auto mode = Modes::GPU;

    auto original = MakeData<double>({ width });
    auto transformed = original;

    HartleyTransform<double> ht(width, 0, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

// ---- 2D ----
// ------- CPU ------
TEST(FDHT, Test2D_Small_CPU)
{
    size_t width = 4;
    size_t height = 4;
    auto mode = Modes::CPU;

    auto original = MakeData<double>({ height, width });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

TEST(FDHT, Test2D_Large_CPU)
{
    size_t width = 256;
    size_t height = 256;
    auto mode = Modes::CPU;

    auto original = MakeData<double>({ height, width });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

TEST(FDHT, Test2D_Large_Different_Sizes_CPU)
{
    size_t width = 128;
    size_t height = 4096;
    auto mode = Modes::CPU;

    auto original = MakeData<double>({ height, width });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

// ------- GPU ------
TEST(FDHT, Test2D_Small_GPU)
{
    SKIP_IF_NO_CUDA();
    size_t width = 4;
    size_t height = 4;
    auto mode = Modes::GPU;

    auto original = MakeData<double>({ height, width });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

TEST(FDHT, Test2D_Large_GPU)
{
    SKIP_IF_NO_CUDA();
    size_t width = 256;
    size_t height = 256;
    auto mode = Modes::GPU;

    auto original = MakeData<double>({ height, width });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

/* developing
TEST(FDHT, Test2D_Large_Different_Sizes_GPU)
{
    SKIP_IF_NO_CUDA();
    size_t width = 128;
    size_t height = 4096;
    auto mode = Modes::GPU;

    auto original = MakeData<double>({ height, width });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, 0, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}*/

// ---- 3D ----
// ------- CPU ------
TEST(FDHT, Test3D_Small_CPU)
{
    size_t width = 4;
    size_t height = 4;
    size_t depth = 4;
    auto mode = Modes::CPU;

    auto original = MakeData<float>({ height, width, depth });
    auto transformed = original;

    HartleyTransform<float> ht(width, height, depth, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

TEST(FDHT, Test3D_Large_CPU)
{
    size_t width = 256;
    size_t height = 256;
    size_t depth = 256;
    auto mode = Modes::CPU;

    auto original = MakeData<double>({ height, width, depth });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, depth, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

TEST(FDHT, Test3D_Large_Different_Sizes_CPU)
{
    size_t width = 128;
    size_t height = 256;
    size_t depth = 512;
    auto mode = Modes::CPU;

    auto original = MakeData<double>({ height, width, depth });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, depth, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

// ------- GPU ------
TEST(FDHT, Test3D_Small_GPU)
{
    SKIP_IF_NO_CUDA();
    size_t width = 4;
    size_t height = 4;
    size_t depth = 4;
    auto mode = Modes::GPU;

    auto original = MakeData<double>({ height, width, depth });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, depth, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

TEST(FDHT, Test3D_Large_GPU)
{
    SKIP_IF_NO_CUDA();
    size_t width = 256;
    size_t height = 256;
    size_t depth = 256;
    auto mode = Modes::GPU;

    auto original = MakeData<double>({ height, width, depth });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, depth, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}

/* developing
TEST(FDHT, Test3D_Large_Different_Sizes_GPU)
{
    SKIP_IF_NO_CUDA();
    size_t width = 128;
    size_t height = 256;
    size_t depth = 512;
    auto mode = Modes::GPU;

    auto original = MakeData<double>({ height, width, depth });
    auto transformed = original;

    HartleyTransform<double> ht(width, height, depth, mode);
    ht.ForwardTransform(transformed.data());
    ht.InverseTransform(transformed.data());

    ExpectVectorsNear(original, transformed);
}*/
