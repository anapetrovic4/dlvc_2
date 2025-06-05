from abc import ABCMeta, abstractmethod
import torch

"""
4. 
Performance measures are defined here. 
class SegMetrics inherits abstract class PerformanceMeasure and all its 
abstract methods which need to be defined in SegMetrics
"""

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes

        self.reset()

    def reset(self) -> None: # -> None: is type hinting; function will work without it too, since it doesn't change code behaviour
        '''
        Resets the internal state.
        '''
        ## TODO implement

        self.num_classes = len(self.classes)
        self.intersections = torch.zeros(self.num_classes, dtype=torch.float64) # Exact matches for each class; True Positive
        self.unions = torch.zeros(self.num_classes, dtype=torch.float64) # Total area which prediction and ground truth cover together


    def update(self, prediction: torch.Tensor, # "prediction:" is also type hint. 
               target: torch.Tensor) -> None: # parameter is a place in a function which recieves a value
                                              # argument is value which is passed to that place
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

       ##TODO implement
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Mismatch in batch size between prediction and target!")
        if prediction.ndim != 4 or target.ndim != 3: # ndim returns numbers of dimensions
            raise ValueError("Prediction must be (b, c, h, w), Target must be (b, h, w)!")
        preds = torch.argmax(prediction, dim=1) # (b, h, w)
        ignore_mask = (target != 255) # ignore 255 pixels

        for cls in range(self.num_classes):
            cls_pred = (preds == cls) & ignore_mask
            cls_target = (target == cls) & ignore_mask

            intersection = (cls_pred & cls_target).sum().item() # sum() calculates all values of a tensor and returns a scalar, while item() extracts the result
            union = (cls_pred | cls_target).sum().item()

            self.intersections[cls] += intersection
            self.unions[cls] += union
   

    def __str__(self): # use print(train_metric) or print(val_metric) so that __str__ can be called automatically
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        ##TODO implement
        return f"mIoU: {self.mIoU():.4f}"          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        ##TODO implement

        # IoU = TP / (TP + FP + FN)
        # mIoU = 1 / n (IoU1 + IoU2 + ... + IoUn); n - number of target classes

        valid = self.unions > 0
        if valid.sum() == 0:
            return 0.0 
        
        ious = torch.zeros(self.num_classes)
        print("ious dtype:", ious.dtype)
        print("result dtype:", (self.intersections[valid] / self.unions[valid]).dtype)

        ious[valid] = (self.intersections[valid] / self.unions[valid]).to(ious.dtype) # cast to get the same value types

        return ious[valid].mean().item()






