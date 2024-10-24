# Paper: Dual trustworthy mechanism for illness classification with multi-modality data

# News: This work has been accepted at the 2023 IEEE International Conference on Data Mining Workshops (ICDMW)!

## Data Introduction：
######    1. id： Student ID<br>
######    2. text： University comments. Code is used to generate embeddings for text data using the MPNet (Masked and Permuted) model.<br> Here's an overview of what the code does:<br>It initializes the MPNet tokenizer and model using the 'microsoft/mpnet-base' pre-trained model.<br> It loops over each row in the dataset starting from the second row (index 1).<br> It retrieves the text data from the second column of the current row, tokenizes it, encodes it, and generates the embeddings using the MPNet model. The resulting embeddings are converted to a numpy array.<br>
######    3. Image: The code  generated image embeddings using the ResNet18 model and save them in an Excel file. <br> Here's an overview of what the code does:<br>It defines a function called images_to_vector that takes a list of image paths as input. The function preprocesses the images, loads the ResNet18 model, feeds the images through the model, and computes the mean of the outputs to obtain image embeddings. The image embeddings are converted to a numpy array.<br>It sets the path where the images are located.<br>It loops over each folder in the specified path.<br>For each folder, it retrieves the image paths within the folder and calls the images_to_vector function to obtain the image embeddings.<br>It retrieves the corresponding row from the existing Excel file based on the folder name and writes the image embeddings to the corresponding column in the new Excel file.<br>Every 1000 iterations, it prints the progress.<br>
######    4. label: The grade is assessed by combining the scores of practical teachers and college students (Elo rating system is A, B, C)<br>
######    5. Structure: Merge textbook analysis, learning situation analysis, teaching objectives, teaching priorities, teaching methods, teaching tools,    teaching processes, and teaching reflections into a list format<br>
######    6. class： subject<br>

# Citation

```
@inproceedings{wang2023dual,
  title={Dual trustworthy mechanism for illness classification with multi-modality data},
  author={Wang, Qing and Zhu, Jia and Pan, Changfan and Shi, Jianyang and Meng, Chaojun and Guo, Hanghui},
  booktitle={2023 IEEE International Conference on Data Mining Workshops (ICDMW)},
  pages={356--362},
  year={2023},
  organization={IEEE}
}
```

