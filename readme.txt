Introduction: 
Random Multiparty Perturbation (RMP) allows each participant to perturb his/her tabular data by passing the data through a nonlinear function, and projecting the data to a lower dimension using a participant-specific random matrix. 
Our randomisation-based scheme perturbs data in two stages: the first, nonlinear stage thwarts Bayesian estimation attacks, whereas the second, linear stage resists independent component analysis attacks. For the nonlinear perturbation stage, a new non-linear function called the “repeated Gompertz” function is proposed. The function is designed to condition the pdf of the perturbed data to protect both anomalous and normal data records. Our scheme is assessed in terms of its recovery resistance to the maximum a priori (MAP) estimation attack.

For anomaly detection, a stacked denoising autoencoder (DAE) is used. The hyperparameters of the autoencoder are set based on the best performance on validation set. Feature values in each dataset are normalised to [0, 1] and merged with 5% anomalous records, which are distributed between [0, 0.05] or [0.95, 1]. Anomalies are identified by the autoencoder based on the mean absolute error (MAE) between the inputs and outputs of the training records. According to three sigma rule, a well-known measure for anomaly detection, the reconstruction error is expected to be Gaussian distributed, hence 99.73% of the error values are expected to be within the threshold \mu(e) + 3\sigma(e). An error value larger than the threshold is unlikely and is identified as an anomaly.

How to run:
To reproduce the PrivacyTest result(repeated Gompertz+participant-specific uniform random matrix) for purely Gaussian dataset under maximum a priori (MAP) estimation attack,\cite{lyu2016improved}:
% 9=two_Gompertz+RP; 1=MAP estimation; 0=recovers normal points, 1=recover outliers
runPrivacyTest('Gaussian', 9, 1, 0);
runPrivacyTest('Gaussian', 9, 1, 1);
To reproduce the PrivacyTest result(repeated Gompertz+random projection matrix) for purely Gaussian dataset under maximum a priori (MAP) estimation attack,\cite{lyu2017privacy},\cite{lyu2018privacy}:
% 9=two_Gompertz+RP; 1=MAP estimation; 0=recovers normal points, 1=recover outliers
runPrivacyTest_DKE('Gaussian', 9, 1, 0);
runPrivacyTest_DKE('Gaussian', 9, 1, 1);


To compute the MAE for anomaly detection, feed the original or two-stage perturbed data X to DAE model by calling: 
[result,result_test]=DAE(X).

Requirements:
Matlab
KDE toolbox: https://www.ics.uci.edu/~ihler/code/kde.html
If error occurs during compiling KDE on matlab, try the following solution:
file mex/cpp/BallTreeDensityClass.cpp, replace line 470 with:
type = (BallTreeDensity::KernelType)(int)mxGetScalar(mxGetField(structure,0,"type"));

Remember to cite the following papers if you use any of the code:
@inproceedings{lyu2016improved,
  title={An improved scheme for privacy-preserving collaborative anomaly detection},
  author={Lyu, Lingjuan and Law, Yee Wei and Erfani, Sarah M and Leckie, Christopher and Palaniswami, Marimuthu},
  booktitle={Pervasive Computing and Communication Workshops (PerCom Workshops), 2016 IEEE International Conference on},
  pages={1--6},
  year={2016},
  organization={IEEE}
}
@inproceedings{lyu2017privacy,
  title={Privacy-Preserving Collaborative Deep Learning with Application to Human Activity Recognition},
  author={Lyu, Lingjuan and He, Xuanli and Law, Yee Wei and Palaniswami, Marimuthu},
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
  pages={1219--1228},
  year={2017},
  organization={ACM}
}
@article{lyu2018privacy,
  title={Privacy-preserving collaborative fuzzy clustering},
  author={Lyu, Lingjuan and Bezdek, James C and Law, Yee Wei and He, Xuanli and Palaniswami, Marimuthu},
  journal={Data \& Knowledge Engineering},
  year={2018},
  publisher={Elsevier}
}