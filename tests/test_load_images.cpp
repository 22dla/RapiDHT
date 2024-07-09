#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>

// ������������ ���� ��� �������� ������������� �������� �������
namespace fs = std::filesystem;

int main() {
    // ���� � ����� � �������������
    std::string folderPath = "D:\\work\\images\\testImages";

    // ������ ��� �������� ���� �����������
    std::vector<cv::Mat> images;

    // ����� ���� ������ � ��������� �����
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        // ��������, ��� ���� �������� ������������ (�����������)
        if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".bmp")) {
            // �������� �����������
            cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);

            // �������� ���������� ��������
            if (img.empty()) {
                std::cerr << "������ �������� �����������: " << entry.path().string() << std::endl;
                continue;
            }

            // ���������� ����������� � ������
            images.push_back(img);
        }
    }

    // ��������, ��� ����������� ���� ���������
    if (images.empty()) {
        std::cerr << "��� ����������� ��� ��������." << std::endl;
        return -1;
    }

    // �������� 3D ������� �����������
    // �������: ���������� ����������� x ������ x ������ x ���������� �������
    int numImages = images.size();
    int height = images[0].rows;
    int width = images[0].cols;
    int channels = images[0].channels();

    // �������� 3D ������� ��� �������� ���� �����������
    cv::Mat image3D(numImages, height, width, CV_8UC1);

    for (int i = 0; i < numImages; ++i) {
        // ����������� ������ ����������� � 3D ������
        images[i].copyTo(image3D.row(i));
    }

    // ������ � ��� ���� 3D ������ ����������� � ���������� image3D
    std::cout << "��������� " << numImages << " ����������� �������� " << height << "x" << width << std::endl;

    return 0;
}
