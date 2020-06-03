def find_max_path(root, max_so_far):
    if not root:
        return 0
    left_result = find_max_path(root.left, max_so_far)
    right_result = find_max_path(root.right, max_so_far)

    current = max(root.val, max(root.val + left_result, root.val + right_result))
    max_so_far[0] = max(max(left_result + right_result + root.val, current), max_so_far[0])
    return current


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        max_so_far = [float("-inf")]
        find_max_path(root, max_so_far)
        return int(max_so_far[0])


tr = TreeNode(-10)
tr.left = TreeNode(9)
tr.left.left = TreeNode(5)
tr.left.right = TreeNode(10)

tr.right = TreeNode(20)
tr.right.left = TreeNode(15)
tr.right.right = TreeNode(7)

print(Solution().maxPathSum(tr))
