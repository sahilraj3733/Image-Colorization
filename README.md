 # 🎨 Image Colorization Using Conditional GAN (cGAN)
 
This project uses a Conditional GAN (cGAN) model to automatically colorize grayscale images. The generator is based on the U-Net architecture, while the discriminator follows the PatchGAN structure. The model learns to predict realistic color channels (a, b) from grayscale (L) images in the Lab color space, trained on the CelebA dataset.


# 📌 Project Highlights


- ✅ Model: Conditional GAN (cGAN) with U-Net Generator and PatchGAN Discriminator

- 🎯 Dataset: CelebA (aligned & cropped facial images)

- 🎨 Color Space: Lab (L: input, a+b: predicted)

- 📉 Loss at Epoch 20:

     - Generator Loss: 5.81

     - Discriminator Loss: 0.63

- 🖥️ Deployed via Streamlit for real-time grayscale-to-color translation


# 🧠 Model Architecture

## U-Net Generator
A symmetric encoder-decoder network with skip connections that preserve spatial details during colorization.

![Screenshot 2025-04-19 142201](https://github.com/user-attachments/assets/b9044d83-5ab4-4beb-ad98-8af94286a582)

## PatchGAN Discriminator

Focuses on local patches instead of full images to encourage high-frequency detail in output.
![10278_2022_696_Fig1_HTML](https://github.com/user-attachments/assets/d24fa63e-b485-47c5-ac15-e15d958b8178)

# 🔄 Input & Output Flow

- **Input**: Grayscale image (L channel)

- **Generator Output**: Predicted a and b color channels

- **Final Output**: Combined (L + ab) → converted to RGB using Lab to RGB conversion



# 📸 Example Results

![Screenshot 2025-04-20 174904](https://github.com/user-attachments/assets/e589eec3-6151-4900-ae94-61f6eca859ae)

![Screenshot 2025-04-20 180334](https://github.com/user-attachments/assets/20198abb-6389-4eaf-9b45-d0d04326ddb2)



