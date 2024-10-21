import numpy as np
def random_samples_and_remaining(img, img_gt_lab, img_gt_ind, num_samples):
    unique_classes = np.unique(img_gt_lab)  # 获取所有类别
    train_indices = []
    remaining_indices = np.arange(img_gt_lab.shape[0])  # 初始为所有样本的索引
    random_indices_log = []  # 用于记录生成的随机索引
    for cls in unique_classes:
        class_indices = np.where(img_gt_lab == cls)[0]  # 获取该类的所有样本索引
        if len(class_indices) >= num_samples:
            random_index = np.random.choice(class_indices, num_samples, replace=False)  # 随机选择num_samples个索引
        else:
            random_index = np.random.choice(class_indices, num_samples, replace=True)  # 如果样本不足，允许重复
        train_indices.extend(random_index)  # 将索引加入训练集
        random_indices_log.append((cls, random_index))  # 记录该类的随机索引
        remaining_indices = np.setdiff1d(remaining_indices, random_index)  # 从剩余索引中删除训练集索引
    train_indices = np.array(train_indices)
    test_indices = remaining_indices  # 剩余的索引即为测试集
    # 获取训练集和测试集
    train_img = img[train_indices, :]
    train_labels = img_gt_lab[train_indices]
    train_indices_final = img_gt_ind[train_indices]
    test_img = img[test_indices, :]
    test_labels = img_gt_lab[test_indices]
    test_indices_final = img_gt_ind[test_indices]

    # 返回训练集、测试集和记录的随机索引
    return train_img, train_labels, train_indices_final, test_img, test_labels, test_indices_final, random_indices_log