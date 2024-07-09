#include <iostream>
#include <vector>
#include <filesystem>
#include <utilities.h>

// ������������ ���� ��� �������� ������������� �������� �������
namespace fs = std::filesystem;

int main() {
	// ���� � ����� � �������������
	std::string folderPath = "D:\\work\\images\\testImages";

    // �������� ����������� � �������� 3D �������
    std::vector<std::vector<std::vector<uint8_t>>> image3D = loadImagesTo3DArray(folderPath);

    // �������� ���������� �������� �����������
    if (image3D.empty()) {
        std::cerr << "Loading is fail." << std::endl;
        return -1;
    }

	// ������ � ��� ���� 3D ������ ����������� � ���������� image3D
	std::cout << "Loaded " << image3D.size() << " images with sizes "
		<< image3D.front().size() << "x" << image3D.front().front().size() << std::endl;

	return 0;
}

