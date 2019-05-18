# Illumination-Estimation-With-Residue-Networks
This works extends [SfSNet](https://github.com/bhushan23/SfSNet-PyTorch)

## Problem Statement
SfSNet generated shading is generated using Normal and Spherical Harmonics which does not
captures the full illumination details. Due to geometric imperfection and spherical harmonics
inaccuracy, generated shading is not near to perfect. Later, shading is used along with albedo to
reconstruct the image. Following image shows traditional shading model which is based on Normal and
Spherical Harmonics. 

![Traditional Illumination Model](report/images/method/old_illumination.png)

## Solution
We propose to introduce new shading layer to capture more flexible and comprehensive illumination
effect that is not modelled by 27 dimensional spherical harmonics like SfSNet. We capture this
representation directly using image features and residual block being used in SfSNet for albedo and
normal. Later, we add this representation into SfSNet based generated shading. This shading layer is
residue of illumination missed by SfSNet. Figure 6 shows new illumination model we are proposing

![Proposed Illumination Model](report/images/method/new_illumination.png)

Following is the flow as compared to SfSNet we are adding.
![SfSNet model](report/images/arch/our_arch.png)

Illumination model based on Normal and Spherical Harmonics is not accurate and cannot model the illumination estimation.
Hence, we are proposing new illumination model, which works on top of Spherical Harmonics and Normal based 
Measure issue with reconstructin faces using Shading, Albedo, Normal and Spherical Harmonics is little variations in normal and spherical harmonics causing different illumination.

Estimating Normal and Spherical Harmonics with slight error causes different illumination and expected.
In order to overcome this problem, we propose new shading model (let's say residue) which works along with traditional illumination model and captures the illumination details missed by traditional model.

We propose different methods to capture this residue using following methods.
We also propose new illumination models, first to correct shading using latent lighting and second to generating shading directly without use of Spherical Harmonics.

### Shading Correcting Network
1_Shading_Correcting works on priciple built on top of tradition shading generation using normal and spherical harmonics
We generate Shading and then rectify it using latent ligthing generated from Normal, Albedo and Image features.

#### Architecture
![Architecture](report/images/out/1_shading_correcting/arch.png)
#### Interpolcation on real data
![Interpolation on real data](report/images/out/1_shading_correcting/res_1_shading_correcting_real.png)
#### Interpolation on synthetic data
![Interpolation on synthetic data](report/images/out/1_shading_correcting/res_1_shading_correcting_syn.png)
#### More training samples
![More training samples](report/images/out/1_shading_correcting/train_out.png)

### Latent Shading Generation
2_Latent_Shading_Gen works on principle to avoid generation of Spherical Harmonics and instead generate shading directly using Normal, Albedo and Image features. And then use this latent lighting along with Normal to generate shading

#### Architecture
![Architecture](report/images/out/2_latent_shading_gen/arch.png)
#### Interpolation on real data
![Interpolation on real data](report/images/out/2_latent_shading_gen/res_2_latent_shading_real.png)
#### Interpolation on synthetic data
![Interpolation on synthetic data](report/images/out/2_latent_shading_gen/res_2_latent_shading_gen_syn.png)
#### More training samples
![More training samples](report/images/out/2_latent_shading_gen/train_out.png)

### 3_Shading_Residue
This method predicts shading residue and adds into traditional shading model.
New shading residue model is capturing details missed by traditional shading model.

#### Architecture
![Architecture](report/images/out/3_shading_res/arch.png)
#### Interpolation on real data
![Interpolation on real data](report/images/out/3_shading_res/res_3_shading_res_real.png)
#### Interpolation on synthetic data
![Interpolation on synthetic data](report/images/out/3_shading_res/res_3_shading_residual_syn.png)
#### More training samples
![More training samples](report/images/out/3_shading_res/train_out.png)

### 4_Shading_Albedo_Residue
This method is next step of Shading residue method which add manual control by adding predicted
residue into Shading and subtracting from Albedo. We can refer this method to be baesd on
Robinhood principle- which takes from albedo and gives to shading.

![Architecture](report/images/out/4_shading_albedo_res/arch.png)
#### Interpolation on real data
![Interpolation on real data](report/images/out/4_shading_albedo_res/res_4_shading_albedo_res_real.png)
#### Interpolation on synthetic data
![Interpolation on synthetic data](report/images/out/4_shading_albedo_res/res_4_shading_albedo_res_syn.png)
#### More training samples
![More training samples](report/images/out/4_shading_albedo_res/train_out.png)

### 5_Shading_Residual_Albedo_GAN
Problem with shading residue method is domain gap of Synthetic and Real albedo being generated. Hence, we use GAN to generate  robust and albedo from synthetic domain which has real ground truth.

#### Architecture
![Architecture](report/images/out/5_gan_albedo/arch.png)
#### Interpolation on real data
![Interpolation on real data](report/images/out/5_gan_albedo/res_gan_real.png)
#### Interpolation on synthetic data
![Interpolation on synthetic data](report/images/out/5_gan_albedo/res_gan_real.png)
#### More training samples
![More training samples](report/images/out/5_gan_albedo/train_out.png)
#### More training samples with No Albedo loss
due to lack of albedo loss, intensity of albedo is pushed into residue
![More training samples with No albedo Loss](report/images/out/5_gan_albedo/train_out_no_albedo_loss.png)

### 6_Shading_Resdiual_GAN_Separate_Train
In method 5, we train GAN based albedo generation and residue network along with each other. In this method, We first train GAN based albedo generation and then we fix GAN based method and then only train residue network.

### 7_Shading_Residual_GAN_train_2
Method 7 is slight variation of method 6 with different Discriminator

## Results samples comparison across the experiments
![Real Sample 1](report/images/out/compare_all/real_1.png)
![Real Sample 2](report/images/out/compare_all/real_2.png)
![Real Sample 3](report/images/out/compare_all/real_3.png)
![Real Sample 4](report/images/out/compare_all/real_4.png)
![Synthetic Sample 1](report/images/out/compare_all/syn_1.png)

