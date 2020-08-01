

def get_label_names():
    '''
    Label info --
        0: Background
        1-20: Objects,
        255: Null (at the borders and on difficult objects)
    '''
    label_mapping = {0:'BACKGROUND',
                     1:'Aeroplane', 2:'Bicycle', 3:'Bird', 4:'Boat', 5:'Bottle',
                     6:'Bus', 7:'Car', 8:'Cat', 9:'Chair', 10:'Cow',
                     11:'Dining table', 12:'Dog', 13:'Horse', 14:'Motor bike', 15:'Person',
                     16:'Potted plant', 17:'Sheep', 18:'Sofa', 19:'Train', 20:'TV monitor',
                     255:'NULL'}

    return label_mapping


def get_color_palette(label_mask):
    rgb_mask = label_mask.convert('RGB')
    return rgb_mask