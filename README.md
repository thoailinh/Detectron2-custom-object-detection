# Detectron2-custom-object-detection

in this repository, I used model zoo of Detectron2 to pre-trained custom object detections (Vietnamese administrative documents)

# Download Data

Download Data from google drive:

- Data raw(https://drive.google.com/file/d/1H6Ka-lz1SyXMOLlajXwgPxvYbhnJ0RhR/view?usp=sharing)

- Data after processing(https://drive.google.com/file/d/1vQT8I76h6EKv3ru9hD7a9sWQGyiOPn2e/view?usp=sharing)

Format data after processing:

- Images folder contain images

- Annotations folder contain txt files

```
# number + ".txt"
x1,y1,x2,y2,x3,y3,x4,y4,id
x1,y1,x2,y2,x3,y3,x4,y4,id
...
...
```

# Conversion of Dataset to COCO Format

Run file processing_data_for_document_layout_analysis.ipynb on google colab
 
# Train 

Run file object_detection_by_detectron2.ipynb to training. You can use models below to have best result follow your data.

#### License

All models available for download through this document are licensed under the
[Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).

### COCO Object Detection Baselines

#### Faster R-CNN:
<!--
(fb only) To update the table in vim:
1. Remove the old table: d}
2. Copy the below command to the place of the table
3. :.!bash

./gen_html_table.py --config 'COCO-Detection/faster*50*'{1x,3x}'*' 'COCO-Detection/faster*101*' --name R50-C4 R50-DC5 R50-FPN R50-C4 R50-DC5 R50-FPN R101-C4 R101-DC5 R101-FPN X101-FPN --fields lr_sched train_speed inference_speed mem box_AP
-->


<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: faster_rcnn_R_50_C4_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_C4_1x.yaml">R50-C4</a></td>
<td align="center">1x</td>
<td align="center">0.551</td>
<td align="center">0.102</td>
<td align="center">4.8</td>
<td align="center">35.7</td>
<td align="center">137257644</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_DC5_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml">R50-DC5</a></td>
<td align="center">1x</td>
<td align="center">0.380</td>
<td align="center">0.068</td>
<td align="center">5.0</td>
<td align="center">37.3</td>
<td align="center">137847829</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/model_final_51d356.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml">R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.210</td>
<td align="center">0.038</td>
<td align="center">3.0</td>
<td align="center">37.9</td>
<td align="center">137257794</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml">R50-C4</a></td>
<td align="center">3x</td>
<td align="center">0.543</td>
<td align="center">0.104</td>
<td align="center">4.8</td>
<td align="center">38.4</td>
<td align="center">137849393</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml">R50-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.378</td>
<td align="center">0.070</td>
<td align="center">5.0</td>
<td align="center">39.0</td>
<td align="center">137849425</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml">R50-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.209</td>
<td align="center">0.038</td>
<td align="center">3.0</td>
<td align="center">40.2</td>
<td align="center">137849458</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_C4_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml">R101-C4</a></td>
<td align="center">3x</td>
<td align="center">0.619</td>
<td align="center">0.139</td>
<td align="center">5.9</td>
<td align="center">41.1</td>
<td align="center">138204752</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_DC5_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml">R101-DC5</a></td>
<td align="center">3x</td>
<td align="center">0.452</td>
<td align="center">0.086</td>
<td align="center">6.1</td>
<td align="center">40.6</td>
<td align="center">138204841</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml">R101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.286</td>
<td align="center">0.051</td>
<td align="center">4.1</td>
<td align="center">42.0</td>
<td align="center">137851257</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/metrics.json">metrics</a></td>
</tr>
<!-- ROW: faster_rcnn_X_101_32x8d_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml">X101-FPN</a></td>
<td align="center">3x</td>
<td align="center">0.638</td>
<td align="center">0.098</td>
<td align="center">6.7</td>
<td align="center">43.0</td>
<td align="center">139173657</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/metrics.json">metrics</a></td>
</tr>
</tbody></table>

#### RetinaNet:
<!--
./gen_html_table.py --config 'COCO-Detection/retina*50*' 'COCO-Detection/retina*101*' --name R50 R50 R101 --fields lr_sched train_speed inference_speed mem box_AP
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: retinanet_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml">R50</a></td>
<td align="center">1x</td>
<td align="center">0.205</td>
<td align="center">0.041</td>
<td align="center">4.1</td>
<td align="center">37.4</td>
<td align="center">190397773</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_50_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml">R50</a></td>
<td align="center">3x</td>
<td align="center">0.205</td>
<td align="center">0.041</td>
<td align="center">4.1</td>
<td align="center">38.7</td>
<td align="center">190397829</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/metrics.json">metrics</a></td>
</tr>
<!-- ROW: retinanet_R_101_FPN_3x -->
 <tr><td align="left"><a href="configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml">R101</a></td>
<td align="center">3x</td>
<td align="center">0.291</td>
<td align="center">0.054</td>
<td align="center">5.2</td>
<td align="center">40.4</td>
<td align="center">190397697</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/metrics.json">metrics</a></td>
</tr>
</tbody></table>


#### RPN & Fast R-CNN:
<!--
./gen_html_table.py --config 'COCO-Detection/rpn*' 'COCO-Detection/fast_rcnn*' --name "RPN R50-C4" "RPN R50-FPN" "Fast R-CNN R50-FPN" --fields lr_sched train_speed inference_speed mem box_AP prop_AR
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">prop.<br/>AR</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: rpn_R_50_C4_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/rpn_R_50_C4_1x.yaml">RPN R50-C4</a></td>
<td align="center">1x</td>
<td align="center">0.130</td>
<td align="center">0.034</td>
<td align="center">1.5</td>
<td align="center"></td>
<td align="center">51.6</td>
<td align="center">137258005</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_C4_1x/137258005/model_final_450694.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_C4_1x/137258005/metrics.json">metrics</a></td>
</tr>
<!-- ROW: rpn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/rpn_R_50_FPN_1x.yaml">RPN R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.186</td>
<td align="center">0.032</td>
<td align="center">2.7</td>
<td align="center"></td>
<td align="center">58.0</td>
<td align="center">137258492</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_FPN_1x/137258492/metrics.json">metrics</a></td>
</tr>
<!-- ROW: fast_rcnn_R_50_FPN_1x -->
 <tr><td align="left"><a href="configs/COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml">Fast R-CNN R50-FPN</a></td>
<td align="center">1x</td>
<td align="center">0.140</td>
<td align="center">0.029</td>
<td align="center">2.6</td>
<td align="center">37.8</td>
<td align="center"></td>
<td align="center">137635226</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fast_rcnn_R_50_FPN_1x/137635226/model_final_e5f7ce.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fast_rcnn_R_50_FPN_1x/137635226/metrics.json">metrics</a></td>
</tr>
</tbody></table>

# Predict 

Run file predict.py to Inference from the Dataset using the trained model and save image with bouding boxes.
