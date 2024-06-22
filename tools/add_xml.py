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

def create_xml(name,classes,root_dir,data):
    root = ET.Element('annotation')

    # 创建 source 元素及其子元素
    source = ET.SubElement(root, 'source')
    filename = ET.SubElement(source, 'filename')
    filename.text = f'{name}.tif'
    origin = ET.SubElement(source, 'origin')
    origin.text = 'Optical'

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

    # 创建 objects 元素及其子元素
    objects = ET.SubElement(root, 'objects')

    # 创建第一个 object 元素及其子元素
    object1 = ET.SubElement(objects, 'object')
    coordinate1 = ET.SubElement(object1, 'coordinate')
    coordinate1.text = 'pixel'
    type1 = ET.SubElement(object1, 'type')
    type1.text = 'rectangle'
    description1 = ET.SubElement(object1, 'description')
    description1.text = 'None'
    possibleresult1 = ET.SubElement(object1, 'possibleresult')
    name1 = ET.SubElement(possibleresult1, 'name')
    name1.text = classes
    points1 = ET.SubElement(object1, 'points')
    point1 = ET.SubElement(points1, 'point')
    point1.text = f"{data[2]},{data[3]}"
    point2 = ET.SubElement(points1, 'point')
    point2.text = f"{data[4]},{data[5]}"
    point3 = ET.SubElement(points1, 'point')
    point3.text = f"{data[6]},{data[7]}"
    point4 = ET.SubElement(points1, 'point')
    point4.text = f"{data[8]},{data[9]}"
    point5 = ET.SubElement(points1, 'point')
    point5.text = f"{data[2]},{data[3]}"
    indent(root)

    tree = ET.ElementTree(root)

    # 将 XML 树写入文件
    tree.write(f'{root_dir}/{name}.xml', encoding='utf-8', xml_declaration=True)

def add_xml(filename,classes,root_dir,data):
    tree = ET.parse(f'{root_dir}/{filename}.xml')
    root = tree.getroot()

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
    objects = root.find('objects')
    objects.append(new_object)
    indent(root)

    # 保存修改后的 XML 文件
    tree.write(f'{root_dir}/{filename}.xml', encoding='utf-8', xml_declaration=True)

def main(work_dir,output_dir):
    # 解析现有的 XML 文件
    file_pattern = os.path.join(work_dir,"**.txt")
    file_list = glob.glob(file_pattern,recursive=True)
    # output_dir = "/data5/laiping/tianzhibei/output_path"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(len(file_list))
    img_id = set()
    for file in file_list:
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()  # 去除行尾的换行符和空白字符
                line_split = line.split(" ")
                name = line_split[0]
                fbasename = os.path.splitext(os.path.basename(file))[0]
                classes = fbasename.split("_")[1]
                if classes == "A320":
                    classes = "A320/321"
                if name not in img_id:
                    img_id.add(name)
                    create_xml(name,classes,output_dir,line_split)
                else:
                    # print(name)
                    add_xml(name,classes,output_dir,line_split)
                # break
        # break
    print(len(img_id))