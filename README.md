# 用 RESNET50 訓練垃圾分類器

## *資料集*
參考並結合 kaggle 上的兩個資料集並依照台灣的回收方式進行分類

|  | 資料內容 |
| :----: | :----: |
| Trash | 包裝紙、口罩、尿布、塑膠袋、牙刷|
| Plastic | 各種常見塑膠容器 |
| Cardboard | 硬紙容器 (飲料杯、便當盒……)、紙箱 |
| Paper | 書報雜誌、印刷品 |
| Metal | 鐵鋁製品，錫箔 |
| Glass | 玻璃製品 |
| 3C | 耳機、線材、電腦相關硬件 |
| Battery | 鋰、鹼性、水銀等電池 |
| Biological | 廚餘 |
| Shoes | 各式鞋子 (不限材質) |
| Clothes | 各式布製衣褲 |
</br>
Kaggle 資料集 </br>
https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2 </br>
https://www.kaggle.com/datasets/raijincheng/rethink-recycle-dataset

## *訓練處理*
![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202025-09-15%20215448.png)

## *訓練結果*
|  | Acc | Loss | Overview |
| :----: | :----: | :----: | :----: |
| FC層無DROPOUT/</br>無分層訓練 |![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/no_Drop_no_SplitTrain/no_Drop_no_SplitTrain_acc.png)|![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/no_Drop_no_SplitTrain/no_Drop_no_SplitTrain_loss.png)|![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/no_Drop_no_SplitTrain/no_Drop_no_SplitTrain.png)|
| FC層DROPOUT/</br>無分層訓練 |![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/Drop_no_SplitTrain/Drop_no_SplitTrain_acc.png)|![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/Drop_no_SplitTrain/Drop_no_SplitTrain_loss.png)|![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/Drop_no_SplitTrain/Drop_no_SplitTrain.png)|
| FC層無DROPOUT/</br>分層訓練 |![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/no_Drop_SplitTrain/no_Drop_SplitTrain_acc.png)|![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/no_Drop_SplitTrain/no_Drop_SplitTrain_loss.png)|![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/no_Drop_SplitTrain/no_Drop_SplitTrain.png)|
| FC層DROPOUT/</br>分層訓練 |![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/Drop_SplitTrain/Drop_SplitTrain_acc.png)|![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/Drop_SplitTrain/Drop_SplitTrain_loss.png)|![image](https://github.com/yorick9453/Trash-classification-model/blob/main/runs/Drop_SplitTrain/Drop_SplitTrain.png)|
## *使用方式*
直接運行 *inference.py* 即可在本機端 (*localhost:5000*) 看到分類網頁，將欲分類的垃圾放在電腦攝像頭前並按下 *拍攝照片* 按鈕後模型即開始分類，並將結果顯示在網頁下方


