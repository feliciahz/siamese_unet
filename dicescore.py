import os
import numpy as np
from PIL import Image
import csv

def DiceScore(imgPredArr, imgLabelArr):
    imgPredArr1D = imgPredArr.flatten()
    imgLabelArr1D = imgLabelArr.flatten()
    arrSum = imgPredArr1D + imgLabelArr1D
    two_TP = sum(arrSum[arrSum == 2])
    DiceScore_denominator = sum(arrSum)
    DiceScore = two_TP / DiceScore_denominator
    return DiceScore


files = os.listdir(os.path.join(folderPath, 'masks'))

diceScores = []
for file in files:
    file_name = file.split(".")[0]
    predFileName = f"{file_name}.jpg"

    predFilePath = os.path.join(folderPath, 'predict_unet', predFileName)
    predImage = Image.open(predFilePath)
    predArr = np.array(predImage)
    predArr = predArr / 255.
    predArr[predArr > 0.5] = 1
    predArr[predArr <= 0.5] = 0

    labelFilePath = os.path.join(folderPath, 'masks', file)
    labelImage = Image.open(labelFilePath)
    labelImage = labelImage.resize(predImage.size)
    labelArr = np.array(labelImage)
    labelArr = labelArr / 255.
    labelArr[labelArr > 0.5] = 1
    labelArr[labelArr <= 0.5] = 0

    diceScore = DiceScore(predArr, labelArr)
    diceScores.append(diceScore)
    print(diceScore)

with open('dice_scores.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Dice Score'])
    for score in diceScores:
        writer.writerow([score])