from atss_core.structures.rboxlist_ops import rboxlist_ml_nms
from atss_core.structures.rotated_bbox import RotatedBoxList

def select_over_all_levels(self, boxlists, is_rotated):
    num_images = len(boxlists)  # N
    results = []
    for i in range(num_images):
        # multiclass nms
        result = rboxlist_ml_nms(boxlists[i], self.nms_thresh)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.fpn_post_nms_top_n > 0:  # 100
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                number_of_detections - self.fpn_post_nms_top_n + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        results.append(result)
    return results

def prepare_for_coco_rotated_detection(predictions, dataset):
    import pycocotools.mask as mask_util

    coco_results = []
    import mmcv
    t = mmcv.load('results.pkl')
    t
    
    for image_id, prediction in tqdm(enumerate(t)):
        original_id = dataset.id_to_img_map[image_id]

        prediction = prediction[0]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        polys = prediction[:, : -1]
        scores = prediction[:, -1].tolist()
        labels = [1] * len(scores)

        boxlists = RotatedBoxList(polys, (int(image_width), int(image_height)), mode="poly")
        select_over_all_levels(boxlists, True)

        rles = [
            mask_util.frPyObjects([poly], image_height, image_width)
            for poly in polys
        ]
        for rle in rles:
            rle[0]["counts"] = rle[0]["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle[0],
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results