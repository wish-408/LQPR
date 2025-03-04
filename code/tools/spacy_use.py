import matplotlib.pyplot as plt
import random
import spacy
from spacy import displacy

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.weights = []


def generate_random_tree(depth, max_children):
    if depth == 0:
        return None
    root = TreeNode(random.randint(1, 100))
    num_children = random.randint(1, max_children)
    for _ in range(num_children):
        child = generate_random_tree(depth - 1, max_children)
        if child:
            weight = random.randint(1, 10)
            root.children.append(child)
            root.weights.append(weight)
    return root


def get_tree_layout(root, x = 0, y = 0, dx = 1, positions = {}):
    if root:
        positions[root] = (x, y)
        num_children = len(root.children)
        if num_children > 0:
            child_dx = dx / num_children
            for i, child in enumerate(root.children):
                new_x = x - dx / 2+(i + 0.5) * child_dx
                positions = get_tree_layout(child, new_x, y - 1, dx, positions)
    return positions


def draw_tree(positions):
    # plt.figure(figsize=(10, 10))
    plt.plot(0.6,-5, color='blue')
    for node in positions:
        x, y = positions[node]
        plt.text(x, y, str(node.value), ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='k', boxstyle='round'))
        for i, child in enumerate(node.children):
            x1, y1 = positions[child]
            weight = node.weights[i]
            # 使用quiver函数绘制有向边
            dx = x1 - x
            dy = y1 - y
            
            if child.value == "." or child.value == "in":
                plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1.05, color='k', width=0.002, headwidth=5,
                       headlength=8)
                
            else:
                plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1.25, color='k', width=0.002, headwidth=5,
                       headlength=8)
            mid_x = (x + x1) / 2
            mid_y = (y + y1) / 2
            # 向右边偏移一点
            offset = 0.02  
            plt.text(mid_x + offset, mid_y, str(weight), ha='left', va='center', fontsize = 7)

def get_dep_tree(node):
    root_node = TreeNode(node.text)
    for child in node.children:
        child_node = get_dep_tree(child)
        weight = child.dep_
        root_node.children.append(child_node)
        root_node.weights.append(weight)
    return root_node

    

# 生成一个随机深度为3，每个节点最多3个子节点的树
# random_tree = generate_random_tree(3, 3)
# positions = get_tree_layout(random_tree)
# draw_tree(random_tree, positions)
# plt.axis('off')
# plt.show()



nlp = spacy.load("en_core_web_sm")
doc = nlp("System shall let customers register on the website in under 5 minutes.")
for token in doc:
    if token.dep_ == "ROOT":
        root_node = token
        break
    
random_tree = get_dep_tree(root_node)
positions = get_tree_layout(root = random_tree, x = 1, y = 1)
for key in positions:
    print(key.value)
    print(positions[key])
    print("--------------------------")
    
draw_tree(positions)
plt.axis('off')
plt.show()
    
# for token in doc:
#     print(token.text, token.dep_, token.head.text, token.head.pos_,
#             [child for child in token.children])
# displacy.serve(doc, style="dep", options={"compact" : True})