# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import pdb
from maskrcnn_benchmark.layers.set_loss import generalized_box_iou, box_iou

class HungarianMatcherCustom(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, special = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.norm = nn.Softmax(-1)
        self.special = special
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1)  # [batch_size * num_queries, num_classes]
        # out_prob_bg = 1 - out_prob
        # out_prob = torch.cat([out_prob_bg, out_prob], dim = 1)

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_bbox = targets["pred_boxes"].flatten(0, 1)  # [batch_size * num_target_boxes, 4]
        tgt_prob = targets["pred_logits"].flatten(0, 1)  # [batch_size * num_target_boxes, num_classes]
        # tgt_prob_bg = 1 - tgt_prob
        # tgt_prob = torch.cat([tgt_prob_bg, tgt_prob], dim = 1)
       
        # Compute the soft-cross entropy between the predicted token alignment and the GT one for each box
        # import pdb
        
        cost_class = out_prob - tgt_prob.transpose(0,1)
        cost_class = cost_class.abs()
        

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the giou cost betwen boxes
        # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost_giou, _ = box_iou(out_bbox, tgt_bbox)
        cost_giou = -cost_giou

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        C_class = cost_class
        C_class = C_class.view(bs, num_queries, -1).cpu()

        C_bbox = cost_bbox
        C_bbox = C_bbox.view(bs, num_queries, -1).cpu()
        #C[torch.isnan(C)] = 0.0
        #C[torch.isinf(C)] = 0.0
        #print(C)
        
        sizes = [tgt_bbox.size(0)] # assum b = 1
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]


        assignment = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        # calculate the total cost; 
        assignment = assignment[0]
        C = C[0]
        C_class = C_class[0]
        C_bbox = C_bbox[0]
        
        cost = 0
        selected_entries = []
        cost_class = 0
        cost_bbox = 0
        cost_matched_box = 0
        
        
        if self.special: # calculate the difference between boxes
            for first_index, second_index in zip(assignment[0], assignment[1]): 
                if -C[first_index, second_index] > 0.5:
                    cost += C_class[first_index, second_index]
                    selected_entries.append(C[first_index, second_index])
                    cost_class += C_class[first_index, second_index]
                    cost_bbox += C_bbox[first_index, second_index]
        else:
            for first_index, second_index in zip(assignment[0], assignment[1]): 
                cost += C[first_index, second_index]
                selected_entries.append(C[first_index, second_index])
                cost_class += C_class[first_index, second_index]
                cost_bbox += C_bbox[first_index, second_index]
        print(selected_entries, cost)

        return cost, len(selected_entries), selected_entries, cost_class, cost_bbox