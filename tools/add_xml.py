import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import glob
def indent(elem, level=0):
    # 添加元素的缩进
    indent_size = 4
    i = "\n" + level * indent_size * " "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + indent_size * " "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def add_xml(filename,classes,root_dir,data):
    tree = ET.parse(f'{root_dir}/{filename}.xml')
    root = tree.getroot()
    objects = root.find('objects')
    if objects == None:
        objects = ET.SubElement(root,"objects")
    # 创建新的 object 元素及其子元素
    new_object = ET.Element('object')
    coordinate = ET.SubElement(new_object, 'coordinate')
    coordinate.text = 'pixel'
    type = ET.SubElement(new_object, 'type')
    type.text = 'rectangle'
    description = ET.SubElement(new_object, 'description')
    description.text = 'None'
    possibleresult = ET.SubElement(new_object, 'possibleresult')
    name = ET.SubElement(possibleresult, 'name')
    name.text = classes
    points = ET.SubElement(new_object, 'points')
    point1 = ET.SubElement(points, 'point')
    point1.text = f"{data[2]},{data[3]}"
    point2 = ET.SubElement(points, 'point')
    point2.text = f"{data[4]},{data[5]}"
    point3 = ET.SubElement(points, 'point')
    point3.text = f"{data[6]},{data[7]}"
    point4 = ET.SubElement(points, 'point')
    point4.text = f"{data[8]},{data[9]}"
    point5 = ET.SubElement(points, 'point')
    point5.text = f"{data[2]},{data[3]}"
    # # 将新的 object 元素添加到 objects 元素中
    objects.append(new_object)
    indent(root)

    # 保存修改后的 XML 文件
    tree.write(f'{root_dir}/{filename}.xml', encoding='utf-8', xml_declaration=True)


def create_blank_xml(name,root_dir,img_label):
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    root = ET.Element('annotation')

    # 创建 source 元素及其子元素
    source = ET.SubElement(root, 'source')
    filename = ET.SubElement(source, 'filename')
    filename.text = f'{name}.tif'
    origin = ET.SubElement(source, 'origin')
    origin.text = img_label

    # 创建 research 元素及其子元素
    research = ET.SubElement(root, 'research')
    version = ET.SubElement(research, 'version')
    version.text = '1.0'
    author = ET.SubElement(research, 'author')
    author.text = 'Cyber'
    pluginclass = ET.SubElement(research, 'pluginclass')
    pluginclass.text = 'object detection'
    time = ET.SubElement(research, 'time')
    time.text = '2021-07-21'

    # 创建 size 元素及其子元素
    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = '800'
    height = ET.SubElement(size, 'height')
    height.text = '600'
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    indent(root)

    tree = ET.ElementTree(root)

    # 将 XML 树写入文件
    tree.write(f'{root_dir}/{name}.xml', encoding='utf-8', xml_declaration=True)


def main(work_dir,output_dir):
    # 解析现有的 XML 文件
    file_pattern = os.path.join(work_dir,"**.txt")
    file_list = glob.glob(file_pattern,recursive=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(len(file_list))
    # img_id = set()
    for file in tqdm(file_list):
        data_list = []
        with open(file, 'r') as f:
            for line in tqdm(f):
                line = line.strip()  
                line_split = line.split(" ")
                name = line_split[0]
                fbasename = os.path.splitext(os.path.basename(file))[0]
                classes = " ".join(fbasename.split("_")[1:])
                # import ipdb;ipdb.set_trace()
                # if classes == "A320":
                #     classes = "A320/321"
                add_xml(name,classes,output_dir,line_split)