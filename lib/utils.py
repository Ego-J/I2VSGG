import scipy.io as sio
import numpy as np
import pickle
import copy
from copy import deepcopy
import time
import sys
import os
import torch
import json
import itertools
from itertools import product
from collections import deque, defaultdict
import traceback
from torch import Tensor
import torch.nn.functional as F



def _iou(box1, box2):
    left  = max(box1[0],box2[0])
    right = min(box1[2],box2[2])
    up    = max(box1[1],box2[1])
    down  = min(box1[3],box2[3])

    if left>=right or down<=up:
        return 0
    else:
        S1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        S2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        S_cross = (down-up)*(right-left)
        return S_cross/(S1+S2-S_cross)

objects_list = json.load(open("/media/sda1/chenjin/IVSGG/data/VidOR/objects.json","r"))
predicates_list = json.load(open("/media/sda1/chenjin/IVSGG/data/VidOR/predicates.json","r"))

class VideoRelation(object):
    '''
    Represent video level visual relation instances
    ----------
    Properties:
        s_cid - object class id for subject
        pid - predicate id
        o_cid - object class id for object
        straj - merged trajectory of subject
        otraj - merged trajectory of object
        confs_list - list of confident score
    '''
    def __init__(self, s_cid, pid, o_cid, straj, otraj, fstart,confs=1,idex=0):
        self.s_cid = int(s_cid)
        self.pid = int(pid)
        self.o_cid = int(o_cid)
        self.rel_idex_list = [idex]
        self.straj = straj
        self.otraj = otraj
        self.confs_list = [confs]
        self.fstart = fstart
        self.fend = fstart + 1
        self.objects = objects_list
        self.predicates = predicates_list
    
    def triplet(self):
        return [self.s_cid, self.pid, self.o_cid]
    
    def mean_confs(self):
        return np.mean(self.confs_list)
    
    def both_overlap(self, straj, otraj, iou_thr=0.5):
        s_iou = _iou(self.straj[-1], straj[0])
        o_iou = _iou(self.otraj[-1], otraj[0])
        if s_iou >= iou_thr and o_iou >= iou_thr:
            return True
        else:
            return False

    def extend(self, straj, otraj, confs,idex):
        self.straj.extend(straj)
        self.otraj.extend(otraj)
        self.confs_list.append(confs)
        self.rel_idex_list.append(idex)
        self.fend += 1

    def serialize(self):
        obj = dict()
        obj['triplet'] = [
            self.objects[self.s_cid],
            self.predicates[self.pid],
            self.objects[self.o_cid]
        ]
        obj['score'] = float(self.mean_confs())
        obj['duration'] = [
            int(self.fstart),
            int(self.fend)
        ]
        obj['sub_traj'] = self.straj
        obj['obj_traj'] = self.otraj
        obj['rel_idex'] = self.rel_idex_list
        return obj
        
def generate_static_relation_feat(video_relations,save_path,feat_path):
    for vid, video_relation in video_relations.items():
        print(vid, "is generating video features")
        for pno, predicate_traj in enumerate(video_relation):
            flag = True
            pre_class = predicate_traj['triplet'][1]
            rel_idex = predicate_traj['rel_idex']
            # print("at %d-predicate:"%(pno))
            # print(predicate_traj['triplet'])
            # print('begin:%d end:%d'%(predicate_traj['duration'][0],predicate_traj['duration'][1]))
            # print('rel_idex:',rel_idex)
            j = 0
            video_feats = []
            for i in range(predicate_traj['duration'][0],predicate_traj['duration'][1],1):
                frame_feat_path = feat_path+os.sep+vid+os.sep+str(i)+'.npz'
                # ("vid %s frame_feat_path %s"%(vid,frame_feat_path))
                try:
                    data = np.load(frame_feat_path)
                    rel_index = rel_idex[j]
                    j = j + 1
                except:
                    # traceback.print_exc()
                    # if the frame is filled with neibor frames, then the npz is not saved!
                    j = j + 1
                    continue
                video_feats.append(data['pre_feat'][rel_index])
            video_feats = np.array(video_feats).mean(axis=0)
            save_path_video = save_path+os.sep+pre_class
            if not os.path.exists(save_path_video):
                os.makedirs(save_path_video)
            save_path_video = save_path_video+os.sep+vid+'_'+str(pno)+'.npy'
            # print("save_path_video",save_path_video)
            np.save(save_path_video,video_feats)

def greedy_relational_association(frame_relations, max_traj_num_in_clip=100):
    frame_relations.sort(key=lambda x: int(x[0]))
    video_relation_list = []
    last_modify_rel_list = []
    for i, (index, pred_list) in enumerate(frame_relations):
        fstart = index
        sorted_pred_list = sorted(pred_list, key=lambda x: x[0], reverse=True)
        if len(sorted_pred_list) > max_traj_num_in_clip:
            sorted_pred_list = sorted_pred_list[0:max_traj_num_in_clip]
        # merge
        cur_modify_rel_list = []
        if i == 0:
            for pred_idx, pred in enumerate(sorted_pred_list):
                conf_score = pred[0]
                s_cid, pid, o_cid = pred[1]
                straj = [pred[2][0]]
                otraj = [pred[2][1]]
                rel_idex = pred[3]
                r = VideoRelation(s_cid, pid, o_cid, straj, otraj, fstart, confs=conf_score,idex=rel_idex)
                video_relation_list.append(r)
                cur_modify_rel_list.append(r)
        else:
            for pred_idx, pred in enumerate(sorted_pred_list):
                conf_score = pred[0]
                s_cid, pid, o_cid = pred[1]
                straj = [pred[2][0]]
                otraj = [pred[2][1]]
                rel_idex = pred[3]
                last_modify_rel_list.sort(key=lambda r: r.mean_confs(), reverse=True)
                is_merged = False
                for r in last_modify_rel_list:
                    if pred[1] == r.triplet():
                        if (fstart == r.fend) and r.both_overlap(straj,otraj):
                            r.extend(straj, otraj, conf_score,rel_idex)
                            last_modify_rel_list.remove(r)
                            cur_modify_rel_list.append(r)
                            is_merged = True
                            break
                if not is_merged:
                    r = VideoRelation(s_cid, pid, o_cid, straj, otraj, fstart, confs=conf_score,idex=rel_idex)
                    video_relation_list.append(r)
                    cur_modify_rel_list.append(r)
        last_modify_rel_list = cur_modify_rel_list
    tmp_list = []
    for r in video_relation_list:
        if len(r.straj) >= 10:
            tmp_list.append(r)
    video_relation_list = tmp_list
    return [rel.serialize() for rel in video_relation_list]

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).

    Adopted from https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def viou(traj_1, duration_1, traj_2, duration_2):
    """ compute the voluminal Intersection over Union
    for two trajectories, each of which is represented
    by a duration [fstart, fend) and a list of bounding
    boxes (i.e. traj) within the duration.
    """
    if duration_1[0] >= duration_2[1] or duration_1[1] <= duration_2[0]:
        return 0.
    elif duration_1[0] <= duration_2[0]:
        head_1 = duration_2[0] - duration_1[0]
        head_2 = 0
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    else:
        head_1 = 0
        head_2 = duration_1[0] - duration_2[0]
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    v_overlap = 0
    for i in range(tail_1 - head_1):
        roi_1 = traj_1[head_1 + i]
        roi_2 = traj_2[head_2 + i]
        left = max(roi_1[0], roi_2[0])
        top = max(roi_1[1], roi_2[1])
        right = min(roi_1[2], roi_2[2])
        bottom = min(roi_1[3], roi_2[3])
        v_overlap += max(0, right - left + 1) * max(0, bottom - top + 1)
    v1 = 0
    for i in range(len(traj_1)):
        v1 += (traj_1[i][2] - traj_1[i][0] + 1) * (traj_1[i][3] - traj_1[i][1] + 1)
    v2 = 0
    for i in range(len(traj_2)):
        v2 += (traj_2[i][2] - traj_2[i][0] + 1) * (traj_2[i][3] - traj_2[i][1] + 1)
    return float(v_overlap) / (v1 + v2 - v_overlap)


def eval_detection_scores(gt_relations, pred_relations, viou_threshold):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    # for rel in pred_relations:
    #     print(rel['triplet'], rel['duration'])
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((len(pred_relations))) * -np.inf
    for pred_idx, pred_relation in enumerate(pred_relations):
        ov_max = -float('Inf')
        k_max = -1
        for gt_idx, gt_relation in enumerate(gt_relations):
            if not gt_detected[gt_idx]\
                    and tuple(pred_relation['triplet']) == tuple(gt_relation['triplet']):
                s_iou = viou(pred_relation['sub_traj'], pred_relation['duration'],
                        gt_relation['sub_traj'], gt_relation['duration'])
                o_iou = viou(pred_relation['obj_traj'], pred_relation['duration'],
                        gt_relation['obj_traj'], gt_relation['duration'])
                ov = min(s_iou, o_iou)
                if ov >= viou_threshold and ov > ov_max:
                    ov_max = ov
                    k_max = gt_idx
        if k_max >= 0:
            print(pred_relation['triplet'])
            hit_scores[pred_idx] = pred_relation['score']
            gt_detected[k_max] = True
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_relations), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores


def eval_tagging_scores(gt_relations, pred_relations):
    pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)
    # ignore trajectories
    gt_triplets = set(tuple(r['triplet']) for r in gt_relations)
    pred_triplets = []
    hit_scores = []
    for r in pred_relations:
        triplet = tuple(r['triplet'])
        if not triplet in pred_triplets:
            pred_triplets.append(triplet)
            hit_scores.append(r['score'])
    hit_scores = np.asarray(hit_scores)
    for i, t in enumerate(pred_triplets):
        if not t in gt_triplets:
            hit_scores[i] = -np.inf
    tp = np.isfinite(hit_scores)
    fp = ~tp
    cum_tp = np.cumsum(tp).astype(np.float32)
    cum_fp = np.cumsum(fp).astype(np.float32)
    rec = cum_tp / np.maximum(len(gt_triplets), np.finfo(np.float32).eps)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
    return prec, rec, hit_scores

# def eval_recognition_scores(gt_relations,pred_relations,type):

#     traj_preds = defaultdict(list)
#     for gt_idx, gt_relation in enumerate(gt_relations):
#         for pred_relation in pred_relations:
#             s_iou = viou(pred_relation['sub_traj'], pred_relation['duration'],
#                     gt_relation['sub_traj'], gt_relation['duration'])
#             o_iou = viou(pred_relation['obj_traj'], pred_relation['duration'],
#                     gt_relation['obj_traj'], gt_relation['duration'])
#             ov = min(s_iou, o_iou)
#             if ov>0.9:
#                 traj_preds[gt_idx].append(pred_relation[])


def evaluate_recognition(predictions, rec_nreturns=[1, 5]):
    acc_at_n = {"sub":defaultdict(list),"obj":defaultdict(list),"pre":defaultdict(list),"rel":defaultdict(list)}
    objects = {1:defaultdict(list),5:defaultdict(list)}
    for video in predictions.keys():
        print(video)
        for triplet in predictions[video]:
            sub_pred = np.argsort(-triplet["sub_score"])[:10]
            sub_correct = np.equal(sub_pred,np.array([triplet["triplet"][0]]*sub_pred.size))+0
            obj_pred = np.argsort(-triplet["obj_score"])[:10]
            obj_correct = np.equal(obj_pred,np.array([triplet["triplet"][2]]*obj_pred.size))+0
            pre_pred = np.argsort(-triplet["pre_score"])[:10]
            pre_correct = np.equal(pre_pred,np.array([triplet["triplet"][1]]*pre_pred.size))+0
            for nre in rec_nreturns:
                acc_at_n["sub"][nre].append(sub_correct[:nre].sum())
                acc_at_n["obj"][nre].append(obj_correct[:nre].sum())
                acc_at_n["pre"][nre].append(pre_correct[:nre].sum())
                for c in range(1,16):
                    if triplet["triplet"][0] == c:
                        objects[nre][c].append(sub_correct[:nre].sum())
                    if triplet["triplet"][2] == c:
                        objects[nre][c].append(obj_correct[:nre].sum())    
            acc_at_n["rel"][1].append(sub_correct[0]*obj_correct[0]*pre_correct[0])
    for nre in rec_nreturns:
        acc_at_n["sub"][nre] = np.mean(acc_at_n["sub"][nre])
        acc_at_n["obj"][nre] = np.mean(acc_at_n["obj"][nre])
        acc_at_n["pre"][nre] = np.mean(acc_at_n["pre"][nre])
        for c in range(1,16):
             objects[nre][c] = np.mean(objects[nre][c])
             print("object #{} acc@{} : {}".format(c,nre,objects[nre][c]))
    acc_at_n["rel"][1] = np.mean(acc_at_n["rel"][1])
    print('subject recognition accurancy@1: {}'.format(acc_at_n["sub"][1]))
    print('subject recognition accurancy@5: {}'.format(acc_at_n["sub"][5]))
    print('object recognition accurancy@1: {}'.format(acc_at_n["obj"][1]))
    print('object recognition accurancy@5: {}'.format(acc_at_n["obj"][5]))
    print('predicate recognition accurancy@1: {}'.format(acc_at_n["pre"][1]))
    print('predicate recognition accurancy@5: {}'.format(acc_at_n["pre"][5]))
    print('relationship recognition accurancy@1: {}'.format(acc_at_n["rel"][1]))
    return acc_at_n


def evaluate(prediction, gt="./data/VidOR/video_annotations.json", viou_threshold=0.5,
        det_nreturns=[50, 100], tag_nreturns=[1, 5, 10]):
    """ evaluate visual relation detection and visual 
    relation tagging.
    """
    if type(prediction) == str:
        prediction = json.load(open(prediction,"r"))

    video_ap = dict()
    tot_scores = defaultdict(list)
    tot_tp = defaultdict(list)
    prec_at_n = defaultdict(list)
    tot_gt_relations = 0
    groundtruth = json.load(open(gt,"r"))
    print('Number of videos in ground truth: {}'.format(len(groundtruth.keys())))
    print('Number of videos in prediction: {}'.format(len(prediction.keys())))
    print('Computing average precision AP over {} videos...'.format(len(groundtruth)))
    for vid, gt_relations in groundtruth.items():
        if len(gt_relations)==0:
            continue
        predict_relations = prediction.get(vid, [])
        tot_gt_relations += len(gt_relations)


        #######
        # for rel in predict_relations:
        #     if rel["triplet"][0] == "adult" or rel["triplet"][0] == "child":
        #         rel["triplet"][0] = "person"
        #     if rel["triplet"][2] == "adult" or rel["triplet"][2] == "child":
        #         rel["triplet"][2] = "person"
        # for rel in gt_relations:
        #     if rel["triplet"][0] == "adult" or rel["triplet"][0] == "child":
        #         rel["triplet"][0] = "person"
        #     if rel["triplet"][2] == "adult" or rel["triplet"][2] == "child":
        #         rel["triplet"][2] = "person"
        #######
        

        # compute average precision and recalls in detection setting
        det_prec, det_rec, det_scores = eval_detection_scores(
                gt_relations, predict_relations, viou_threshold)
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for nre in det_nreturns:
            cut_off = min(nre, det_scores.size)
            # ?????????????????
            tot_scores[nre].append(det_scores[:cut_off])
            tot_tp[nre].append(tp[:cut_off])
        # compute precisions in tagging setting
        tag_prec, _, _ = eval_tagging_scores(gt_relations, predict_relations)
        for nre in tag_nreturns:
            cut_off = min(nre, tag_prec.size)
            # ?????????????????
            if cut_off > 0:
                prec_at_n[nre].append(tag_prec[cut_off - 1])
            else:
                prec_at_n[nre].append(0.)
    # calculate mean ap for detection
    mean_ap = np.mean(list(video_ap.values()))
    # calculate recall for detection
    rec_at_n = dict()
    print(tot_gt_relations)  
    for nre in det_nreturns:
        scores = np.concatenate(tot_scores[nre])
        tps = np.concatenate(tot_tp[nre])
        sort_indices = np.argsort(scores)[::-1]
        tps = tps[sort_indices]
        cum_tp = np.cumsum(tps).astype(np.float32)
        print(cum_tp[-1])
        rec = cum_tp / np.maximum(tot_gt_relations, np.finfo(np.float32).eps)
        rec_at_n[nre] = rec[-1]
    # calculate mean precision for tagging
    mprec_at_n = dict()
    for nre in tag_nreturns:
        #print(len(prec_at_n[nre]))
        mprec_at_n[nre] = np.mean(prec_at_n[nre])
    # print scores
    print('detection mean AP (used in challenge): {}'.format(mean_ap))
    print('detection recall@50: {}'.format(rec_at_n[50]))
    print('detection recall@100: {}'.format(rec_at_n[100]))
    print('tagging precision@1: {}'.format(mprec_at_n[1]))
    print('tagging precision@5: {}'.format(mprec_at_n[5]))
    print('tagging precision@10: {}'.format(mprec_at_n[10]))
    return mean_ap, rec_at_n, mprec_at_n


def association(frame_relations):

    invalid_num = 4
    max_num_per_video = 200
    video_relations = {}
    cnt = 0
    for vid in frame_relations:
        pred = frame_relations[vid]
        pred.sort(key=lambda x: int(x[0]))
        pred_mask = [0 if len(pred[i][1])==0 else -1 for i in range(len(pred))]
        if -1 not in pred_mask:
            print(vid+" is empty!!!")
            continue
        tmp_mask = [-1 for i in range(len(pred))]
        for i in range(len(pred_mask)):
            if pred_mask[i] == 0:

                j = i - 1
                while j >=0 and pred_mask[j]==0:
                    j -= 1
                if j < 0 :
                    left = 0
                else:
                    left = i - j

                j = i + 1
                while j <len(pred_mask) and pred_mask[j]==0:
                    j += 1
                if j >= len(pred) :
                    right = 0
                else:
                    right = j - i

                if right==0 or (left>0 and left<=right):
                    tmp_mask[i] = i-left
                elif left==0 or (right>0 and left>right):
                    tmp_mask[i] = i+right
        pred_mask = tmp_mask
        for i in range(len(pred_mask)):
            if pred_mask[i] >= 0:
                if i<invalid_num:
                    start = 0
                    end = i+invalid_num
                elif i > len(pred_mask)-invalid_num-1:
                    start = i-invalid_num
                    end = len(pred_mask)-1
                else:
                    start = i - invalid_num
                    end = i + invalid_num
                invalid = True
                for j in range(start,end+1):
                    if pred_mask[j] == -1:
                        invalid = False
                if invalid:
                    pred_mask[i] = -2

        for i in range(len(pred_mask)):
            if pred_mask[i] > -1:
                pred[i][1] = pred[pred_mask[i]][1]

        video_relations[vid] = greedy_relational_association(pred)
        video_relations[vid].sort(key=lambda x: x['score'], reverse=True)
        video_relations[vid] = video_relations[vid][:max_num_per_video]
        print(str(cnt),vid,"association done!","count:",len(video_relations[vid]))
        cnt+=1
    return video_relations


def alignment(frame_recognitions, gt_relations = "./data/video_annotations.json", max_traj_num_in_clip=100):

    with open(gt_relations, 'r') as fp:
        gt = json.load(fp)
    print('Number of videos in ground truth: {}'.format(len(gt.keys())))
    print('Number of videos in prediction: {}'.format(len(frame_recognitions.keys())))
    objects = ['_background','airplane','ball','bear','bicycle','bus','car','cat','dog','elephant','horse','motorcycle','person','skateboard','sofa','train']
    predicates = json.load(open("./data/predicates.json","r"))
    video_relations = {}
    video_recognitions = {}
    #for video in gt.keys():
        # print(video)
        # if video not in frame_recognitions.keys():
        #     continue
        # traj_data = {}
        # for triplet in gt[video]:
        #     stid = triplet["subject_tid"]
        #     otid = triplet["object_tid"]
        #     start = triplet["object_tid"]["duration"][0]
        #     end = triplet["object_tid"]["duration"][1]
        #     if (stid,otid,start,end) not in traj_data:
        #         sub_score_mean = np.zeros((0,16))
        #         obj_score_mean = np.zeros((0,16))
        #         pre_score_mean = np.zeros((0,89))
        #         for fno in range(start,end):
        #             pair_no = frame_recognitions[video][fno]["tids"].index([stid,otid])
        #             sub_score_mean = np.vstack((sub_score_mean,frame_recognitions[video][fno]["sub_scores"][pair_no]))
        #             obj_score_mean = np.vstack((obj_score_mean,frame_recognitions[video][fno]["obj_scores"][pair_no]))
        #             pre_score_mean = np.vstack((pre_score_mean,frame_recognitions[video][fno]["pre_scores"][pair_no]))
        #         sub_score_mean = sub_score_mean.mean(axis=0)
        #         obj_score_mean = obj_score_mean.mean(axis=0)
        #         pre_score_mean = pre_score_mean.mean(axis=0)
        #         traj_data[(stid,otid,start,end)] = {"sub_pred":sub_score_mean,"obj_pred":obj_score_mean,"pre_pred":pre_score_mean,
        #                                            "sub_gt":objects.index(triplet["triplet"][0]),"obj_gt":objects.index(triplet["triplet"][2]),
        #                                            "pre_gt":[predicates[triplet["triplet"][1]]]}
        #     else:
        #         traj_data[(stid,otid,start,end)]["pre_gt"].append(predicates[triplet["triplet"][1]])
        #     video_recognitions[video].append(traj_data)

    return video_relations, video_recognitions

def recognition_output(vrd_data):
    if len(vrd_data["boxes"])<=1:
        # print("Nothing!")
        return None,None,None,None
    sub_scores = vrd_data["sub_scores"]
    sub_scores[:,0] = 0.0
    obj_scores = vrd_data["obj_scores"]
    obj_scores[:,0] = 0.0
    pre_scores = vrd_data["rel_scores"].data.cpu().numpy()
    pre_scores += np.log(0.5*(vrd_data["rel_so_prior"]+1.0/15))
    tids = vrd_data["tids"]
    
    return sub_scores, obj_scores, pre_scores, tids

def detection_output(vrd_data):

    if len(vrd_data["bboxes"])<=1:
        # print("Nothing!")
        return None,None,None,None,None
    ixs = vrd_data["ixs"]
    ixo = vrd_data["ixo"]
    boxes = vrd_data["bboxes"]
    classes = vrd_data["classes"]
    confs = vrd_data["scores"]
    rel_score = vrd_data["rel_score"] 
    rel_so_prior = vrd_data["rel_so_prior"] 
    num_relations = 26
    tuple_confs_im = []
    rel_idex = []
    rlp_labels_im  = np.zeros((100, 3), dtype = np.float)
    sub_bboxes_im  = np.zeros((100, 4), dtype = np.float)
    obj_bboxes_im  = np.zeros((100, 4), dtype = np.float)
    rel_prob = rel_score.data.cpu().numpy()

    # rel_prob += np.log(0.5*(rel_so_prior+1.0/num_relations))
    # rel_prob = Tensor(rel_prob)
    # rel_prob = F.softmax(rel_prob, dim=1)
    # rel_prob = rel_prob.data.cpu().numpy()

    for i in range(rel_prob.shape[0]):
        #print(rel_prob[i],confs[ixs[i]],confs[ixo[i]])
        rel_prob[i] = rel_prob[i]*confs[ixs[i]]*confs[ixo[i]]
    
    # Select top 100 rel predication among rel_instances*rel_nums, rel_res is the 2-d index
    rel_res = np.dstack(np.unravel_index(np.argsort(-rel_prob.ravel()), rel_prob.shape))[0][:100]
    # print("rel_res:",rel_res.shape)
    for ii in range(rel_res.shape[0]):            
        rel = rel_res[ii, 1]
        tuple_idx = rel_res[ii, 0]
        conf = rel_prob[tuple_idx, rel]
        sub_bboxes_im[ii] = boxes[ixs[tuple_idx]]
        obj_bboxes_im[ii] = boxes[ixo[tuple_idx]]
        rlp_labels_im[ii] = [classes[ixs[tuple_idx]], rel, classes[ixo[tuple_idx]]]
        tuple_confs_im.append(conf)
        rel_idex.append(tuple_idx)
    tuple_confs_im = np.array(tuple_confs_im)
    rel_idex = np.array(rel_idex)
    return rlp_labels_im, tuple_confs_im, sub_bboxes_im, obj_bboxes_im, rel_idex
    
# def detection_output_lod(vrd_data):

#     if len(vrd_data["boxes"])<=1:
#         print("Nothing!")
#         return None,None,None,None,None
#     with open('./data/source_so_prior.pkl', 'rb') as fid:
#         so_prior = np.array(pickle.load(fid,encoding='bytes'))
#     ixs =  vrd_data["ixs"]
#     ixo = vrd_data["ixo"]
#     boxes = vrd_data["boxes"]
#     sub_score = vrd_data["sub_score"]
#     obj_score = vrd_data["obj_score"]
#     pre_score = vrd_data["rel_score"] 
#     num_relations = 89
#     tuple_confs_im = np.zeros(0, dtype = np.float)
#     rlp_labels_im  = np.zeros((0, 3), dtype = np.float)
#     sub_bboxes_im  = np.zeros((0, 4), dtype = np.float)
#     obj_bboxes_im  = np.zeros((0, 4), dtype = np.float)
#     rel_idex = np.zeros(0, dtype = np.int)

#     sub_score = sub_score.data.cpu().numpy()
#     sub_score[:,0] = 0.0
#     obj_score = obj_score.data.cpu().numpy()
#     obj_score[:,0] = 0.0
#     pre_score = pre_score.data.cpu().numpy()
    

#     for tuple_idx in range(pre_score.shape[0]):
#         top_s_ind = np.argsort(-sub_score[tuple_idx])[:10] 
#         top_p_ind = np.argsort(-pre_score[tuple_idx])[:10]
#         top_o_ind = np.argsort(-obj_score[tuple_idx])[:10]
#         score = sub_score[tuple_idx][top_s_ind, None, None] * pre_score[tuple_idx][None, top_p_ind, None] * obj_score[tuple_idx][None, None, top_o_ind]
#         for sid,s in enumerate(top_s_ind):
#             for oid,o in enumerate(top_o_ind):
#                 for pid,p in enumerate(top_p_ind):
#                     score[sid,pid,oid] += so_prior[s-1,o-1,p]
#         top_flat_ind = np.argsort(score, axis = None)[-100:]
#         top_score = score.ravel()[top_flat_ind]
#         top_s, top_p, top_o = np.unravel_index(top_flat_ind, score.shape)
#         rels = np.array([(top_s_ind[top_s[i]], top_p_ind[top_p[i]], top_o_ind[top_o[i]]) for i in range(top_score.size)]).reshape(-1,3)
#         tuple_confs_im = np.append(tuple_confs_im,top_score)
#         rel_idex = np.append(rel_idex,np.array([tuple_idx for i in range(top_score.size)]))
#         rlp_labels_im = np.vstack((rlp_labels_im,rels))
#         sub_bboxes_im = np.vstack((sub_bboxes_im,np.array([boxes[ixs[tuple_idx]] for i in range(top_score.size)]).reshape(-1,4))) 
#         obj_bboxes_im = np.vstack((obj_bboxes_im,np.array([boxes[ixo[tuple_idx]] for i in range(top_score.size)]).reshape(-1,4)))

#     return rlp_labels_im, tuple_confs_im, sub_bboxes_im, obj_bboxes_im, rel_idex]

if __name__ == "__main__":
    frame_detections = json.load(open("./frame_detections_results.json","r"))
    video_detections_static = association(frame_detections)
    evaluate(video_detections_static,'../data/VidOR/video_annotations.json')