import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

# --- 1. 配置路径 ---
# 图像文件夹路径
IMAGE_DIR = '/home/xueyijun/data/Vicy/datasets/20250721/'
# XML文件夹路径
XML_DIR = '/mnt/hdd1/xueyijun/Vicy/object_detection/outputs/20250724/xml_output'
# 输出结果的文件夹路径
OUTPUT_DIR = '/mnt/hdd1/xueyijun/Vicy/object_detection/outputs/20250724/vis'
# 我们只关心的目标类别
TARGET_CLASS = 'ship'
# 绘制框的颜色和线宽
BOX_COLOR = 'red'


def draw_boxes_on_image():
    """
    主函数：读取XML，找到对应图像，绘制'ship'目标的边界框，并保存。
    """
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出文件夹: {OUTPUT_DIR}")

    # 遍历XML文件夹中的所有文件
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
    if not xml_files:
        print(f"警告: 在 '{XML_DIR}' 文件夹中没有找到XML文件。")
        return

    print(f"开始处理 {len(xml_files)} 个XML文件...")

    for xml_file in xml_files:
        xml_path = os.path.join(XML_DIR, xml_file)

        try:
            # --- 2. 解析XML文件 ---
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图像文件名
            image_filename = root.find('source/filename').text
            image_path = os.path.join(IMAGE_DIR, image_filename)

            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                print(f"警告: 找不到XML '{xml_file}' 对应的图像 '{image_filename}'，跳过。")
                continue

            # --- 3. 打开图像并创建绘制对象 ---
            # 使用 with 语句确保文件能被正确关闭
            with Image.open(image_path) as img:
                # 转换为RGB模式，以防是其他格式（如灰度图、P模式等）
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)

                ship_found = False
                # --- 4. 查找所有'object'标签 ---
                for obj in root.findall('objects/object'):
                    # 获取类别名称
                    name_element = obj.find('possibleresult/name')
                    if name_element is None:
                        continue

                    class_name = name_element.text

                    # --- 5. 判断是否为'ship'类别 ---
                    if class_name == TARGET_CLASS:
                        ship_found = True
                        # 提取边界框的点
                        points_list = []
                        points_elements = obj.findall('points/point')
                        for point_elem in points_elements:
                            # 坐标格式为 "x,y"
                            coords = point_elem.text.split(',')
                            x = float(coords[0])
                            y = float(coords[1])
                            points_list.append((x, y))

                        # --- 6. 绘制多边形边界框 ---
                        if len(points_list) > 1:
                            draw.polygon(points_list, outline=BOX_COLOR)

                # --- 7. 保存绘制好的图像 ---
                if ship_found:
                    output_path = os.path.join(OUTPUT_DIR, image_filename)
                    img.save(output_path)
                    print(f"处理完成: 在 '{image_filename}' 上绘制了 'ship' 目标框，并保存到 '{output_path}'")
                else:
                    print(f"信息: 在 '{image_filename}' 中未找到 'ship' 目标，不进行保存。")

        except ET.ParseError:
            print(f"错误: XML文件 '{xml_file}' 解析失败，跳过。")
        except Exception as e:
            print(f"处理 '{xml_file}' 时发生未知错误: {e}")

    print("\n所有文件处理完毕！")


if __name__ == '__main__':
    draw_boxes_on_image()