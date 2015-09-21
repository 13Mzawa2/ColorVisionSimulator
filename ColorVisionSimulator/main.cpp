/****************************************************************
	LUT�ǂݍ��݁��摜�ϊ��v���O����
	Author: Nishizawa,	Date: 2015/08/25
	�K�v�ȃ��C�u�����FOpenCV 3.0 (�摜�����C�s�񉉎Z��)

	�y�@�\�z
	PNG�܂���XML�`����LUT��ǂݍ��݁CLUT�ɏ]���ĉ摜�ϊ����܂��D
	�A��XML�`���͔��ɏd���C�r���ŃI�[�o�[�t���[������
	LUT��S�ēǂݍ��߂Ȃ��ꍇ���������肷��̂Ŕ񐄏��ł��D
*****************************************************************/

#include "OpenCV3Linker.h"

using namespace cv;
using namespace std;

//	LUT�̃A�N�Z�X�p�}�N��
#define BIT(B, G, R) ((B) << 16 | (G) << 8 | (R))

//--------------------------------------------
//	�֐��v���g�^�C�v
//--------------------------------------------
void lookup(Mat &src, Mat &dst, Mat &LUT);

//-------------------------
//	�ǂݍ���LUT�̕ۑ��ꏊ
//-------------------------
Mat LUT_elder70;			//	70�΍���҃V�~�����[�V�����p
Mat LUT_elder80;			//	80�΍���҃V�~�����[�V�����p
Mat LUT_typeP;			//	��^(P)��F�o�V�~�����[�V�����p
Mat LUT_typeD;			//	��^(D)��F�o�V�~�����[�V�����p
Mat LUT_typeT;			//	�O�^(T)��F�o�V�~�����[�V�����p

int main(void)
{
	//------------------------------------------
	//	PNG�`��
	//------------------------------------------
	//	���݂�TickCount
	double f = 1000.0 / cv::getTickFrequency();
	int64 time = cv::getTickCount();

	//	PNG�`��LUT�̓ǂݍ���
	cout << "PNG�`����LUT��ǂݍ���ł��܂�..." << endl;
	cout << "����ғ�����ǂݍ��ݒ�..." << endl;
	LUT_elder70 = imread("data/LUT_elder_70.png");
	LUT_elder80 = imread("data/LUT_elder_80.png");
	cout << "�F�o������ǂݍ��ݒ�..." << endl;
	LUT_typeP = imread("data/LUT_dichromat_typeP.png", CV_LOAD_IMAGE_COLOR);
	LUT_typeD = imread("data/LUT_dichromat_typeD.png", CV_LOAD_IMAGE_COLOR);
	LUT_typeT = imread("data/LUT_dichromat_typeT.png", CV_LOAD_IMAGE_COLOR);
	cout << "�ǂݍ��݂��I�����܂���." << endl;

	//	TickCount�̕ω���[ms]�P�ʂŕ\��
	std::cout << "��������: " << (cv::getTickCount() - time)*f << " [ms]" << std::endl;

	////------------------------------------------
	////	XML�`��
	////	�d���ēǂݍ��߂Ȃ����ǈꉞ�c���Ă���
	////------------------------------------------
	////	���݂�TickCount
	//f = 1000.0 / cv::getTickFrequency();
	//time = cv::getTickCount();

	////	XML�`����LUT��ǂݍ���
	//cout << "XML�`����LUT��ǂݍ���ł��܂�..." << endl;
	////	�����LUT
	//cout << "����ғ�����ǂݍ��ݒ�..." << endl;
	//FileStorage elderData("data/LUT_elder.xml", FileStorage::READ);
	//FileNode calibNode = elderData["calibration"];
	//FileNode gammaNode = calibNode["display_gamma"];
	//FileNode luminanceNode = calibNode["display_luminance"];
	//FileNode xyNode = calibNode["display_xy"];
	//FileNode lutNode = elderData["LUT"];
	//double gamma[3] = {
	//	static_cast<double>(gammaNode["gamma_b"]),
	//	static_cast<double>(gammaNode["gamma_g"]),
	//	static_cast<double>(gammaNode["gamma_r"]) 
	//};
	//double luminance[3] = {
	//	static_cast<double>(gammaNode["luminance_b"]),
	//	static_cast<double>(gammaNode["luminance_g"]),
	//	static_cast<double>(gammaNode["luminance_r"]) 
	//};
	//Point2d xyDisp[3];
	//xyNode["xy_b"] >> xyDisp[0];
	//xyNode["xy_g"] >> xyDisp[1];
	//xyNode["xy_r"] >> xyDisp[2];
	//lutNode["seventy"] >> LUT_elder70;
	//lutNode["eighty"] >> LUT_elder80;
	//elderData.release();
	//~calibNode;
	//~gammaNode;
	//~luminanceNode;
	//~xyNode;
	//~lutNode;
	////	�F�oLUT
	//cout << "�F�o������ǂݍ��ݒ�..." << endl;
	//FileStorage dichromantData("data/LUT_elder.xml", FileStorage::READ);
	//calibNode = dichromantData["calibration"];
	//gammaNode = calibNode["display_gamma"];
	//luminanceNode = calibNode["display_luminance"];
	//xyNode = calibNode["display_xy"];
	//lutNode = dichromantData["LUT"];
	//gamma[0] = static_cast<double>(gammaNode["gamma_b"]);
	//gamma[1] = static_cast<double>(gammaNode["gamma_g"]);
	//gamma[2] = static_cast<double>(gammaNode["gamma_r"]);
	//luminance[0] = static_cast<double>(gammaNode["luminance_b"]);
	//luminance[1] = static_cast<double>(gammaNode["luminance_g"]);
	//luminance[2] = static_cast<double>(gammaNode["luminance_r"]);
	//xyNode["xy_b"] >> xyDisp[0];
	//xyNode["xy_g"] >> xyDisp[1];
	//xyNode["xy_r"] >> xyDisp[2];
	//lutNode["type_p"] >> LUT_typeP;
	//lutNode["type_d"] >> LUT_typeD;
	//lutNode["type_t"] >> LUT_typeT;
	//dichromantData.release();
	//~calibNode;
	//~gammaNode;
	//~luminanceNode;
	//~xyNode;
	//~lutNode;

	//cout << "�ǂݍ��݂��I�����܂����D" << endl;

	//// TickCount�̕ω���[ms]�P�ʂŕ\��
	//std::cout << "��������: " << (cv::getTickCount() - time)*f << " [ms]" << std::endl;

	//	���摜�̓ǂݍ���
	Mat original = imread("img/lovelive.png");
	resize(original, original, Size(320, 240));

	//	LUT�ɂ��ϊ�
	Mat elder70, elder80, typeP, typeD, typeT;
	lookup(original, elder70, LUT_elder70);
	lookup(original, elder80, LUT_elder80);
	lookup(original, typeP, LUT_typeP);
	lookup(original, typeD, LUT_typeD);
	lookup(original, typeT, LUT_typeT);

	//	�\��
	imshow("���摜", original);
	imshow("70��", elder70);
	imshow("80��", elder80);
	imshow("1�^2�F�o", typeP);
	imshow("2�^2�F�o", typeD);
	imshow("3�^2�F�o", typeT);

	waitKey(0);

	destroyAllWindows();

	return 0;
}

//	LUT�ɏ]���ĉ�f�l�̓���ւ�
void lookup(Mat &src, Mat &dst, Mat &LUT)
{
	dst = Mat(src.rows, src.cols, CV_8UC3);
	for (int y = 0; y < dst.rows; y++)
	{
		for (int x = 0; x < dst.cols; x++)
		{
			matB(dst, x, y) = LUT.at<Vec3b>(BIT(matB(src, x, y), matG(src, x, y), matR(src, x, y)), 0)[0];
			matG(dst, x, y) = LUT.at<Vec3b>(BIT(matB(src, x, y), matG(src, x, y), matR(src, x, y)), 0)[1];
			matR(dst, x, y) = LUT.at<Vec3b>(BIT(matB(src, x, y), matG(src, x, y), matR(src, x, y)), 0)[2];
		}
	}
}