## 语义分割：DINOv2 + PSPNet（第二个任务）

使用 DINOv2-small（ViT-S/14，冻结 backbone）+ PSPNet 分割头，在 PASCAL VOC 2012 上训练，
验证集 **mIoU 67.5%**，像素准确率约 95%。

![训练曲线](segregation/dinov2_pspnet_voc_curves.png)

![分割效果](segregation/dinov2_pspnet_voc_prediction.png)

说明：DINOv3 权重需 Meta 授权（gated repo），暂用同系列 DINOv2 验证 pipeline，
后续授权通过后迁移补实验。
