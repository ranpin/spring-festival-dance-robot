import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
from scipy.interpolate import make_interp_spline, interp1d

from .image_folder import ImageFolder


def est_camera(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    h, w = image.shape[:2]
    camera = {
        'img_focal': np.max([h, w]),
        'img_center': np.array([w/2, h/2])
    }
    return camera

def est_calib(images):
    """ Roughly estimate intrinsics by image dimensions """
    # imgfiles = sorted(glob(f'{imagedir}/*.jpg'))
    image = images[0] # cv2.imread(imgfiles[0])

    h0, w0, _ = image.shape
    focal = np.max([h0, w0])
    cx, cy = w0/2., h0/2.
    calib = [focal, focal, cx, cy]
    return calib


def detect_track(images, savedir=None, visualization=False,
                 yolo_thresh=0.10, 
                 bytetrack_thresh=0.25, 
                 bytetrack_match=0.8,
                 bbox_interp=False):

    # Yolo + Bytetrack
    yolo = YOLO("data/yolo11x.pt")
    box_annotator = sv.BoxAnnotator(thickness=5)
    tracker = sv.ByteTrack(track_activation_threshold=bytetrack_thresh,
                            minimum_matching_threshold=bytetrack_match)
    tracks = defaultdict(lambda: defaultdict(list))

    for i in range(len(images)):
        img = images[i]
        if isinstance(img, str):
            img = cv2.imread(images[i])[:,:,::-1]
            
        results = yolo(img, verbose=False, classes=0, conf=yolo_thresh)[0] 
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        for k in range(len(detections)):
            d = detections[k]
            tid = d.tracker_id.item()
            tracks[tid]['masks'].append(np.full(img.shape[:2], False))
            tracks[tid]['bboxes'].append(d.xyxy[0])
            tracks[tid]['frames'].append(i)

        # Save visualization
        if visualization:
            annotated_frame = box_annotator.annotate(img[:,:,::-1].copy(), 
                                                    detections=detections,
                                                    custom_color_lookup=detections.tracker_id
                                                    )
            height, width, layers = annotated_frame.shape
            size = (width, height)

            try:
                out.write(annotated_frame)
            except Exception:
                print('Start writing video ...')
                video_path = f'{savedir}/vis.mp4'
                fourcc, fps = cv2.VideoWriter_fourcc(*'mp4v'), 30
                out = cv2.VideoWriter(video_path, fourcc, fps, size)

    for k in tracks:
        masks = np.stack(tracks[k]['masks'])
        bboxes = np.stack(tracks[k]['bboxes'])
        frames = np.array(tracks[k]['frames'])

        if bbox_interp:
            interp_bboxes, interp_frames, interp_masks = interpolate_bboxes(bboxes, frames, masks, fn='linear')
        else:
            interp_bboxes, interp_frames, interp_masks = bboxes, frames, masks
        
        tracks[k]['track_id'] = k
        tracks[k]['frames'] = interp_frames
        tracks[k]['bboxes'] = interp_bboxes
        tracks[k]['masks'] = interp_masks
        tracks[k]['detected'] = np.sum(interp_masks, axis=(1, 2)) > 1
        
    tracks = recursive_to_dict(tracks)

    if visualization:
        out.release()
        del out

    return tracks


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _detect_people_at(detector, images, i, nms_oks_thresh, det_thresh, score_thresh, height_thresh):
    """Run the person detector + same filtering as keyframe selection at frame i."""
    img_cv2 = cv2.imread(images[i]) if isinstance(images[i], str) else images[i][..., ::-1]
    det_out = detector(img_cv2)
    keypoints = det_out['instances'].pred_keypoints.cpu().numpy()
    scores = det_out['instances'].scores.cpu().numpy()
    boxes = det_out['instances'].pred_boxes.tensor.cpu().numpy()
    if len(scores) == 0:
        return np.zeros((0, 4)), np.zeros((0, keypoints.shape[1] if keypoints.ndim == 3 else 17, 3))
    keep = nms_oks(keypoints.transpose(0, 2, 1), boxes, nms_oks_thresh)
    keypoints, scores, boxes = keypoints[keep], scores[keep], boxes[keep]
    conf = scores > det_thresh
    keypoints, scores, boxes = keypoints[conf], scores[conf], boxes[conf]
    if len(scores) == 0:
        return np.zeros((0, 4)), np.zeros((0, keypoints.shape[1], 3))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    bbox_height = boxes[:, 3] - boxes[:, 1]
    combined = 0.5 * areas / np.max(areas) + 0.5 * scores / np.max(scores)
    idxs = [x for x in np.argsort(combined)[::-1] if combined[x] > score_thresh]
    if len(idxs) > 1:
        largest = bbox_height[np.argmax(bbox_height)]
        idxs = [x for x in idxs if bbox_height[x] > largest * height_thresh]
    return boxes[idxs], keypoints[idxs]


def detect_segment_track_sam(images, out_path, paths_dict, debug_masks, sam2_type, 
                             detector_type='detectron2', filter_ng_points=False, kp_thres=0.1, 
                             num_max_people=10, height_thresh=0.3, score_thresh=0.4, det_thresh=0.5, 
                             bbox_interp=False, nms_oks_thresh=0.5, keyframe_search=False,
                             keyframe_stride=10, keyframe_search_frac=0.15,
                             reacquire=False, reacq_stride=50, reacq_iou=0.3):
    from torch.utils.data import DataLoader
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
    from detectron2.config import get_cfg
    from .detector.sam2_video_predictor import build_sam2_video_predictor
    from .utils_detectron2 import DefaultPredictor
    
    device = 'cuda'
    segm_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', 
                                weights=DeepLabV3_ResNet50_Weights.DEFAULT).to(device)
    segm_model.eval()
    
    segm_dataloader = DataLoader(ImageFolder(images), batch_size=8, shuffle=False, 
                                    num_workers=4 if os.cpu_count() > 4 else os.cpu_count())

    deeplab_masks = []
    for batch in segm_dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
                
        with torch.no_grad():
            output = segm_model(batch['img'])['out']
        mask = (output.argmax(1) == 15).to(torch.float) # the max probability is the person class
        mask = mask.cpu()
        deeplab_masks.append(mask > 0.5)
    deeplab_masks = torch.cat(deeplab_masks, dim=0)
    masks = deeplab_masks.cpu().numpy()
    del deeplab_masks
    
    sam2_registry = {
        'tiny': {
            'checkpoint': paths_dict['sam2'] + '/sam2_hiera_tiny.pt',
            'config': "pipeline/sam2/sam2_hiera_t.yaml",
        },
        'small': {
            'checkpoint': paths_dict['sam2'] + '/sam2_hiera_small.pt',
            'config': "pipeline/sam2/sam2_hiera_s.yaml",
        },
        'base_plus': {
            'checkpoint': paths_dict['sam2'] + '/sam2_hiera_base_plus.pt',
            'config': "pipeline/sam2/sam2_hiera_b+.yaml",
        },
        'large': {
            'checkpoint': paths_dict['sam2'] + '/sam2_hiera_large.pt',
            'config': "pipeline/sam2/sam2_hiera_l.yaml"
        },
    }

    checkpoint = sam2_registry[sam2_type]['checkpoint']
    model_cfg = sam2_registry[sam2_type]['config']
    if not os.path.exists(model_cfg):
        model_cfg = '/code/' + model_cfg
    model_cfg = '/' + os.path.abspath(model_cfg)
    predictor = build_sam2_video_predictor(model_cfg, checkpoint)
    
    if detector_type == 'detectron2':
        # ViTDet
        cfg_path = 'pipeline/detectron2/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
        if not os.path.exists(cfg_path):
            cfg_path = f"/code/{cfg_path}"
        
        import logging
        logging.getLogger().setLevel(logging.WARNING)

        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25
        cfg.MODEL.WEIGHTS = f"{paths_dict['sam2']}/keypoint_rcnn_5ad38f.pkl"
        detector = DefaultPredictor(cfg)
        kp_thres = 0.1
     
    start_frame = -1
    # Group-dance fix: the track count is fixed by whichever frame initializes SAM2.
    # The original code broke at the FIRST frame with detections, so a dancer who
    # overlaps another (NMS OKS) or crouches (height filter) in that one frame is
    # permanently dropped. Instead scan candidate frames and pick the one with the
    # most people, tie-broken by the most spread formation (lowest max pairwise OKS).
    best_cand = None  # (num_people, -max_oks, frame_idx, boxes, scores, keypoints, combined)
    if keyframe_search:
        # SAM2 propagates forward only from the keyframe, so it must be early to cover
        # the whole video. Restrict the search to the first fraction of frames; among
        # those pick most people then most spread. A late keyframe with extra background
        # people would otherwise win on count and leave only a few propagated frames.
        kf_end = max(keyframe_stride + 1, int(len(images) * keyframe_search_frac))
        frame_iter = range(0, kf_end, keyframe_stride)
    else:
        frame_iter = range(len(images))
    for i in frame_iter:
        if isinstance(images[i], str):
            img_cv2 = cv2.imread(images[i])
        elif isinstance(images[i], np.ndarray):
            img_cv2 = images[i][..., ::-1]

        if detector_type == 'detectron2':
            det_out = detector(img_cv2)
            keypoints = det_out['instances'].pred_keypoints.cpu().numpy()
            scores = det_out['instances'].scores.cpu().numpy()
            boxes = det_out['instances'].pred_boxes.tensor.cpu().numpy()

        if len(scores) == 0:
            continue
        keep = nms_oks(keypoints.transpose(0, 2, 1), boxes, nms_oks_thresh)
        keypoints = keypoints[keep]
        scores = scores[keep]
        boxes = boxes[keep]

        conf = scores > det_thresh
        keypoints = keypoints[conf]
        scores = scores[conf]
        boxes = boxes[conf]
        if len(scores) == 0:
            continue

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        bbox_height = boxes[:, 3] - boxes[:, 1]
        normalized_areas = areas / np.max(areas)
        normalized_scores = scores / np.max(scores)
        combined_score = 0.5 * normalized_areas + 0.5 * normalized_scores
        best_idxs = [x for x in np.argsort(combined_score)[::-1] if combined_score[x] > score_thresh]

        if len(best_idxs) > 1:
            largest_person_height = bbox_height[np.argmax(bbox_height)]
            best_idxs = [x for x in best_idxs if bbox_height[x] > largest_person_height * height_thresh]

        if len(best_idxs) == 0:
            continue
        bb = boxes[best_idxs]
        bs = scores[best_idxs]
        bk = keypoints[best_idxs]
        bc = combined_score[best_idxs]
        spread = _max_pairwise_oks(bk, bb)
        cand = (len(best_idxs), -spread, i, bb, bs, bk, bc)
        if best_cand is None or (cand[0], cand[1]) > (best_cand[0], best_cand[1]):
            best_cand = cand
            print(f"[keyframe] frame {i}: {len(best_idxs)} people, max OKS={spread:.3f} (best so far)")
        if not keyframe_search:
            break

    if best_cand is None:
        raise ValueError("No persons detected in any of the images.")
    _, _, start_frame, best_boxes, best_scores, best_keypoints, best_combined_scores = best_cand
    print(f"[keyframe] selected frame {start_frame}: {len(best_boxes)} people")
    print(f"Best person scores: {best_scores}")
    print(f"Best person combined scores: {best_combined_scores}")
            
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if isinstance(images[0], str):
            state = predictor.init_state(
                video_path=os.path.dirname(images[0]),
                async_loading_frames=False,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
        elif isinstance(images[0], np.ndarray):
            state = predictor.init_state(
                video_frames=images,
                async_loading_frames=False,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )

        if debug_masks:
            img_debug = images[start_frame][..., ::-1].copy()
            
        for idx in range(len(best_boxes)):
            pos_kp = best_keypoints[idx][best_keypoints[idx, :, 2] > kp_thres]
            neg_kp = []
            if filter_ng_points:
                for pid in range(len(best_boxes)):
                    if pid != idx:
                        neg_p_kp = best_keypoints[pid][best_keypoints[pid, :, 2] > kp_thres]
                        # check if the keypoints are within the bounding box
                        x1, y1, x2, y2 = best_boxes[idx]
                        mask = (neg_p_kp[:, 0] >= x1) & (neg_p_kp[:, 0] <= x2) & \
                            (neg_p_kp[:, 1] >= y1) & (neg_p_kp[:, 1] <= y2)
                        if mask.any():
                            neg_kp.append(neg_p_kp[mask])
                        
            if len(neg_kp) > 0:
                neg_kp = np.concatenate(neg_kp, axis=0)
                box_p = best_boxes[idx].reshape(-1, 2)
                box_labels = np.array([2, 3])
                points = np.concatenate([pos_kp[:, :2], neg_kp[:, :2], box_p], axis=0)
                point_labels = np.concatenate([np.ones(len(pos_kp)), np.zeros(len(neg_kp)), box_labels]).astype(np.int64)
            else:
                points = np.concatenate([pos_kp[:, :2], best_boxes[idx].reshape(-1, 2)], axis=0)
                point_labels = np.concatenate([np.ones(len(pos_kp)), np.array([2, 3])]).astype(np.int64)
            
            _, out_id, msk = predictor.add_new_points(
                state,
                frame_idx=start_frame,
                obj_id=idx+1,
                labels=point_labels, 
                points=points,
            )
            
            if debug_masks:                
                mask_colors = [
                    (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), 
                    (0, 255, 255), (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0),
                ]
                c = mask_colors[idx % len(mask_colors)]
                
                for pt, l in zip(points, point_labels):
                    pc = c if l != 0 else (0, 0, 0)
                    img_debug = cv2.circle(img_debug, (int(pt[0]), int(pt[1])), 3, pc, -1)
                
                # draw text object id in the middle of the bounding box
                cx, cy = (best_boxes[idx, :2] + best_boxes[idx, 2:]) / 2
                img_debug = cv2.putText(img_debug, f"{idx+1}", (int(cx), int(cy)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                img_debug = cv2.putText(img_debug, f"{idx+1}", (int(cx), int(cy)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                color_mask = np.zeros_like(img_debug)
                msk = (msk[idx, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                color_mask[msk > 0] = c
                img_debug = cv2.rectangle(img_debug, 
                                      (int(best_boxes[idx, 0]), int(best_boxes[idx, 1])), 
                                      (int(best_boxes[idx, 2]), int(best_boxes[idx, 3])), 
                                      c, 2)
                img_v = img_debug.copy() * 0.7 + color_mask * 0.3
                img_debug[msk > 0] = img_v[msk > 0]
                cv2.imwrite(f"{out_path}/init_masks.jpg", img_debug)
        
        video_segments = {}
        track_results = {}

        def _collect(prop_iter):
            for t, out_obj_ids, out_mask_logits in prop_iter:
                if debug_masks:
                    video_segments[t] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                boxes = batched_mask_to_box(out_mask_logits > 0.0)
                for obj_id, box, obj_msk in zip(out_obj_ids, boxes, out_mask_logits):
                    tr = track_results.setdefault(obj_id, {
                        'track_id': obj_id, 'bbox_format': 'x1y1x2y2', 'by_frame': {}})
                    tr['by_frame'][t] = (box[0].cpu().numpy(),
                                         (obj_msk[0] > 0.0).cpu().numpy().astype(np.bool_))

        # propagate the prompts to get masklets throughout the video
        _collect(predictor.propagate_in_video(state))

        # Multi-keyframe re-acquisition: SAM2 forward-only propagation permanently loses
        # dancers who exit the frame. Re-detect at later frames and re-condition the lost
        # object id when it re-appears, then re-propagate from that frame (overwrites >=F).
        if reacquire:
            n_frames = len(images)

            def _last_valid(tr):
                vs = [t for t, (b, m) in tr['by_frame'].items()
                      if (b[2] - b[0]) > 10 and (b[3] - b[1]) > 10]
                return max(vs) if vs else -1

            lost = [oid for oid, tr in track_results.items()
                    if _last_valid(tr) < n_frames - 1 - reacq_stride]
            for oid in sorted(lost, key=lambda o: _last_valid(track_results[o])):
                tr = track_results[oid]
                last = _last_valid(tr)
                lb = tr['by_frame'][last][0]
                lc = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2])
                for F in range(last + reacq_stride, n_frames, reacq_stride):
                    db, dk = _detect_people_at(detector, images, F, nms_oks_thresh,
                                               det_thresh, score_thresh, height_thresh)
                    if len(db) == 0:
                        continue
                    alive = [t2['by_frame'][F][0] for t2 in track_results.values()
                             if F in t2['by_frame']
                             and (t2['by_frame'][F][0][2] - t2['by_frame'][F][0][0]) > 10]
                    unmatched = [j for j in range(len(db))
                                 if all(_box_iou(db[j], ab) < reacq_iou for ab in alive)]
                    if not unmatched:
                        continue
                    j = min(unmatched, key=lambda j: np.linalg.norm(
                        np.array([(db[j][0] + db[j][2]) / 2, (db[j][1] + db[j][3]) / 2]) - lc))
                    pos_kp = dk[j][dk[j, :, 2] > kp_thres]
                    points = np.concatenate([pos_kp[:, :2], db[j].reshape(-1, 2)], axis=0)
                    labels = np.concatenate([np.ones(len(pos_kp)), np.array([2, 3])]).astype(np.int64)
                    predictor.add_new_points(state, frame_idx=F, obj_id=oid,
                                             labels=labels, points=points)
                    _collect(predictor.propagate_in_video(state, start_frame_idx=F))
                    print(f"[reacquire] track {oid} lost after {last}, re-init at frame {F}")
                    break

        for k in track_results.keys():
            ts = sorted(track_results[k]['by_frame'].keys())
            frames = np.array(ts)
            bboxes = np.array([track_results[k]['by_frame'][t][0] for t in ts])
            obj_masks = np.array([track_results[k]['by_frame'][t][1] for t in ts])
            
            bb_heights = bboxes[:, 3] - bboxes[:, 1]
            bb_widths = bboxes[:, 2] - bboxes[:, 0]
            
            # Remove the 0s at the beginning and end of the bb_heights
            non_zero_indices = np.where(np.logical_and(bb_heights > 0, bb_widths > 0))[0]
            if len(non_zero_indices) > 0:
                start_index = non_zero_indices[0]
                end_index = non_zero_indices[-1] + 1
                
                frames = frames[start_index:end_index]
                bboxes = bboxes[start_index:end_index]
                obj_masks = obj_masks[start_index:end_index]
                bb_heights = bb_heights[start_index:end_index]
                bb_widths = bb_widths[start_index:end_index]
                
            # for the remaining interpolated frames, interpolate the bounding boxes
            bboxes = bboxes[np.logical_and(bb_heights > 10, bb_widths > 10)]
            frames = frames[np.logical_and(bb_heights > 10, bb_widths > 10)]
            obj_masks = obj_masks[np.logical_and(bb_heights > 10, bb_widths > 10)]
            
            if len(bboxes) == 0 or len(frames) == 0:
                print(f"No valid bboxes or frames for track {k}")
                continue
            
            if bbox_interp:
                interp_bboxes, interp_frames, interp_masks = interpolate_bboxes(bboxes, frames, obj_masks, fn='linear')
            else:
                interp_bboxes, interp_frames, interp_masks = bboxes, frames, obj_masks
            
            track_results[k]['frames'] = interp_frames
            track_results[k]['bboxes'] = interp_bboxes
            track_results[k]['masks'] = interp_masks
            track_results[k]['detected'] = np.sum(interp_masks, axis=(1, 2)) > 1
            
    # sort the track_results by the number of frames and then take the num_max_people 
    track_results = dict(sorted(track_results.items(), key=lambda x: len(x[1]['frames']), reverse=True))
    keep_track_ids = list(track_results.keys())[:num_max_people]
    track_results = {k: v for k, v in track_results.items() if k in keep_track_ids}
    
    del detector
    del predictor
    del segm_model

    return track_results, masks


def detect_segment_track_sam3(images, out_path, bbox_interp=False, num_max_people=10,
                              prompt_frame=0, prompt_text="person"):
    """SAM3.1 Object Multiplex tracker: text-concept detection + dense tracking.

    Returns (track_results, masks) with the SAME interface as detect_segment_track_sam
    so downstream (ViTPose / SLAM / HPS / world) is unchanged. Unlike SAM2's forward-only
    propagation, the decoupled detector re-finds concept instances, handling dancers who
    exit and re-enter the frame.
    """
    import tempfile
    import inspect
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if isinstance(images[0], np.ndarray):
        frame_dir = os.path.join(out_path, "sam3_frames")
        os.makedirs(frame_dir, exist_ok=True)
        for i, im in enumerate(images):
            cv2.imwrite(os.path.join(frame_dir, f"{i:06d}.jpg"), im[..., ::-1])
        resource = frame_dir
    else:
        resource = os.path.dirname(images[0])

    predictor = build_sam3_multiplex_video_predictor()
    # sam3 main-branch bug: base start_session passes offload_state_to_cpu/video_loader_type
    # which the multiplex tracking init_state signature rejects. Patch at CLASS level
    # (predictor itself has no init_state attribute; it resolves to the tracking mixin).
    from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingWithInteractivity as _C
    _orig_init = _C.init_state
    _sig = inspect.signature(_orig_init)
    def _patched_init(self, *a, **kw):
        kw = {k: v for k, v in kw.items() if k in _sig.parameters}
        return _orig_init(self, *a, **kw)
    _C.init_state = _patched_init
    # T4 (sm75) lacks fp8 flash-attn-3; vitdet imports fa3.flash_attn_func at call time,
    # so swap it for an sdpa fallback (bf16/fp16) that Turing can run.
    import sam3.perflib.fa3 as _fa3
    def _sdpa_attn(q, k, v):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return o.transpose(1, 2)
    _fa3.flash_attn_func = _sdpa_attn
    r = predictor.handle_request(request=dict(type="start_session", resource_path=resource))
    sid = r["session_id"]
    predictor.handle_request(request=dict(type="add_prompt", session_id=sid,
                                          frame_index=prompt_frame, text=prompt_text))

    by_frame = {}
    for resp in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
        t = resp["frame_index"]
        out = resp["outputs"]
        ids = np.asarray(out["out_obj_ids"]).tolist()
        masks_arr = np.asarray(out["out_binary_masks"])
        frame_masks = {}
        for i, oid in enumerate(ids):
            m = masks_arr[i].astype(np.bool_)
            if not m.any():
                continue
            ys, xs = np.where(m)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)
            frame_masks[oid] = (box, m)
        by_frame[t] = frame_masks
    # rebuild per-frame union mask stack
    T = len(images)
    H, W = images[0].shape[:2] if isinstance(images[0], np.ndarray) else cv2.imread(images[0]).shape[:2]
    masks = np.zeros((T, H, W), np.bool_)
    track_results = {}
    for t, fm in by_frame.items():
        u = np.zeros((H, W), np.bool_)
        for oid, (box, m) in fm.items():
            tr = track_results.setdefault(oid, {'track_id': oid, 'bbox_format': 'x1y1x2y2', 'by_frame': {}})
            tr['by_frame'][t] = (box, m)
            u |= m
        masks[t] = u

    for k in track_results.keys():
        ts = sorted(track_results[k]['by_frame'].keys())
        frames = np.array(ts)
        bboxes = np.array([track_results[k]['by_frame'][t][0] for t in ts])
        obj_masks = np.array([track_results[k]['by_frame'][t][1] for t in ts])
        bb_h = bboxes[:, 3] - bboxes[:, 1]
        bb_w = bboxes[:, 2] - bboxes[:, 0]
        good = np.logical_and(bb_h > 10, bb_w > 10)
        frames, bboxes, obj_masks = frames[good], bboxes[good], obj_masks[good]
        if len(frames) == 0:
            print(f"No valid bboxes or frames for track {k}")
            continue
        if bbox_interp:
            ib, iff, im = interpolate_bboxes(bboxes, frames, obj_masks, fn='linear')
        else:
            ib, iff, im = bboxes, frames, obj_masks
        track_results[k]['frames'] = iff
        track_results[k]['bboxes'] = ib
        track_results[k]['masks'] = im
        track_results[k]['detected'] = np.sum(im, axis=(1, 2)) > 1

    track_results = dict(sorted(track_results.items(), key=lambda x: len(x[1]['frames']), reverse=True))
    keep = list(track_results.keys())[:num_max_people]
    track_results = {k: v for k, v in track_results.items() if k in keep}
    del predictor
    return track_results, masks


def recursive_to_dict(obj):
    """Convert a nested structure of defaultdicts (and any normal dicts) into plain dicts."""
    if isinstance(obj, defaultdict):          # or `collections.abc.Mapping`
        return {k: recursive_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):               # keep lists intact, but descend into them
        return [recursive_to_dict(v) for v in obj]
    else:
        return obj 
    
    
def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out


def _max_pairwise_oks(keypoints, boxes):
    """Max pairwise OKS among kept detections (keypoints N×K×3, boxes N×4).

    Lower means the formation is more spread out; used to pick a keyframe where no
    two dancers overlap enough for NMS to mistake them for one person.
    """
    n = len(boxes)
    if n < 2:
        return 0.0
    kps = keypoints.transpose(0, 2, 1)  # N×3×K
    m = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            o = float(compute_oks(kps[i], boxes[i], kps[j:j + 1], boxes[j:j + 1])[0])
            if o > m:
                m = o
    return m


def nms_oks(kp_predictions, rois, thresh):
    """Nms based on kp predictions."""
    scores = np.mean(kp_predictions[:, 2, :], axis=1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(
            kp_predictions[i], rois[i], kp_predictions[order[1:]],
            rois[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    # measure the per-keypoint distance if keypoints visible
    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / e.shape[1]

    return e


def interpolate_bboxes(bboxes, frames, masks, fn='linear'):
    '''
    bboxes: numpy array of shape (len(frames), 4) representing the bounding boxes in x, y, x, y format
    frames: example [0, 1, 2, 8, 9, 10] -> here the frames 3-7 are missing and should be interpolated
    '''
    
    # Create a continuous range of frames
    all_frames = np.arange(frames[0], frames[-1] + 1)
    
    # Interpolate bounding boxes
    if fn == 'spline':
        interp_bboxes = make_interp_spline(frames, bboxes, k=3)(all_frames)
    elif fn == 'linear':
        # Use scipy's interp1d for linear interpolation
        interp_func = interp1d(frames, bboxes, axis=0, kind='linear')
        interp_bboxes = interp_func(all_frames)
    else:
        raise ValueError("Invalid interpolation function. Choose 'spline' or 'linear'.")   
    
    all_masks = np.full([len(all_frames), masks.shape[1], masks.shape[2]], False, dtype=bool)
    indices = frames - all_frames[0]
    all_masks[indices] = masks

    return interp_bboxes, all_frames, all_masks







