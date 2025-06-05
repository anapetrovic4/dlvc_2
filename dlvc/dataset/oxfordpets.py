from collections import namedtuple
from torchvision.datasets import OxfordIIITPet

"""
2.
Attributes and segmentation classes of the dataset are defined and annotated.
"""

OxfordpetsLabels = namedtuple(
        "OxfordpetsLabels",
        ["name", "id"],
    )

"""
Note that the labels are (1,2,3) and pytorch losses etc. work with 0-based labels,
don't forget to subtract 1 from the labels vector,
when using the dataset 
(in the given trainer reference implementation, this is already taken care of)
"""

class OxfordPetsCustom(OxfordIIITPet):
    classes_seg=[
        OxfordpetsLabels("pet",  0,),
        OxfordpetsLabels("background",  1 ),
        OxfordpetsLabels("border",  2 )
    ]