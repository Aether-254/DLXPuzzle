import argparse
import psutil
import os
import time
import gc
import threading
TEST = False


class PuzzlePiece:
    def __init__(self, segments, count=1):
        """
        segments: list of tuples (dy, length)
        count: 该类型块的数量
        """
        self.segments = segments  # [(dy0, len0), (dy1, len1), ...]
        self.count = count
        self.width = len(segments)  # 块的列数


""" 拼图块定义 
# 格式:
# --- col
# OXX | ln
# XXX | ln
# XXO | ln
# (0, 2) (0, 3) (1, 2)
# 即为 PuzzlePiece([(0,2), (0,3), (1,2)], 1)
# 解释:
# PuzzlePiece([(dy, leny), (dy, leny), ...], count)
# x轴是列表索引
"""


def create_test_case():
    pieces = [
        PuzzlePiece([(0, 1)], count=1)  # 1x1方块
    ]
    grid_width, grid_height = 1, 1  # 1x1网格
    return pieces, grid_width, grid_height


def get_valid_placements(piece):
    """
    返回一个piece所有有效的放置位置(x_start, y_base)
    """
    # print("get_valid_placements 被 %s 线程调用了!" % threading.current_thread().name)
    placements = []
    # 计算y_base的范围
    min_y_base = 0
    max_y_base = GRID_HEIGHT - 1

    for dy, length in piece.segments:
        min_y_base = max(min_y_base, -dy)  # y_base + dy >= 0 ⇒ y_base >= -dy
        # y_base + dy + length - 1 <= 13
        max_y_base = min(max_y_base, GRID_HEIGHT - (dy + length - 1))

    if min_y_base > max_y_base:
        return []  # 无有效放置

    # 遍历所有可能的位置
    for x_start in range(GRID_WIDTH):
        for y_base in range(min_y_base, max_y_base + 1):
            placements.append((x_start, y_base))

    return placements


print("State: 2/5 - 定义函数 get_valid_placements 完成")


def get_covered_cells(piece, x_start, y_base):
    """
    返回一个放置位置覆盖的网格单元格列表(x, y)
    """
    # print("get_covered_cells 被 %s 线程调用了!" % threading.current_thread().name)
    covered = []
    for i, (dy, length) in enumerate(piece.segments):
        x = (x_start + i) % GRID_WIDTH  # 环状处理
        for y in range(y_base + dy, y_base + dy + length):
            covered.append((x, y))
    return covered



def build_matrix_and_placements(pieces, GRID_WIDTH, GRID_HEIGHT):
    # 为每个piece类型计算所有放置位置（含180度旋转）
    def rotate_piece(piece):
        # 180度旋转：segments反转，dy变号，保持length不变
        new_segments = [(-dy, length) for dy, length in reversed(piece.segments)]
        return PuzzlePiece(new_segments, piece.count)

    print("State: 3/5 - 计算所有放置位置完成（含旋转）")
    matrix_rows = []
    piece_info = []
    cell_col_offset = 0
    piece_col_offset = GRID_WIDTH * GRID_HEIGHT

    # 对每个拼图块的每个副本，枚举所有放置（原始+旋转），都归属于同一个 piece_col
    for piece_idx, piece in enumerate(pieces):
        piece_start_col = piece_col_offset + sum(p.count for p in pieces[:piece_idx])
        for copy_idx in range(piece.count):
            piece_col = piece_start_col + copy_idx
            # 原始方向
            placements = get_valid_placements(piece)
            for x_start, y_base in placements:
                covered_cells = get_covered_cells(piece, x_start, y_base)
                row_columns = [x * GRID_HEIGHT + y for x, y in covered_cells]
                row_columns.append(piece_col)
                matrix_rows.append(row_columns)
                piece_info.append((piece_idx, copy_idx, x_start, y_base, False))
            # 旋转方向（如果和原始不同）
            rotated_piece = rotate_piece(piece)
            if rotated_piece.segments != piece.segments:
                placements_rot = get_valid_placements(rotated_piece)
                for x_start, y_base in placements_rot:
                    covered_cells = get_covered_cells(rotated_piece, x_start, y_base)
                    row_columns = [x * GRID_HEIGHT + y for x, y in covered_cells]
                    row_columns.append(piece_col)
                    matrix_rows.append(row_columns)
                    piece_info.append((piece_idx, copy_idx, x_start, y_base, True))
    print("State: 4/5 - 矩阵行构建完成")
    return matrix_rows, piece_info


class DLXNode:
    def __init__(self, col_idx=None):
        self.left = self.right = self.up = self.down = self
        self.col = col_idx  # 列头节点
        self.size = 0      # 该列的行数


class DLX:
    def __init__(self, num_cols):
        print("一个 DLX 实例被创建.")
        self.header = DLXNode()
        self.nodes = []
        self.columns = [DLXNode(i) for i in range(num_cols)]
        self.steps = 0
        print(f"DLX 初始化中 - 已定义 {num_cols} 列.")
        # 初始化列头双向链表
        prev = self.header
        for col_node in self.columns:
            col_node.left = prev
            prev.right = col_node
            prev = col_node
        prev.right = self.header
        self.header.left = prev
        print("DLX 初始化中 - 列头链表已连接.")
        # 初始化每个列的上下指针
        for col_node in self.columns:
            col_node.up = col_node.down = col_node
        print("DLX 初始化中 - 列头上下指针已初始化.")

    def add_row(self, row_columns, row_info_idx=None):
        """添加一行到矩阵中，row_info_idx为该行在piece_info中的索引"""
        first_node = None
        for col_idx in row_columns:
            col_node = self.columns[col_idx]
            new_node = DLXNode(col_idx)
            self.nodes.append(new_node)

            # 插入到列中
            new_node.up = col_node.up
            new_node.down = col_node
            col_node.up.down = new_node
            col_node.up = new_node
            col_node.size += 1

            # 连接到行中
            if first_node:
                new_node.left = first_node.left
                new_node.right = first_node
                first_node.left.right = new_node
                first_node.left = new_node
            else:
                first_node = new_node
                first_node.left = first_node.right = first_node
        # 绑定 piece_info 索引到行头节点
        if first_node is not None and row_info_idx is not None:
            first_node.piece_info_idx = row_info_idx

    def cover(self, col_node):
        """覆盖一列"""
        # print("DLX.cover 被 %s 线程调用了!" % threading.current_thread().name)
        col_node.right.left = col_node.left
        col_node.left.right = col_node.right

        row_node = col_node.down
        while row_node != col_node:
            col_node2 = row_node.right
            while col_node2 != row_node:
                col_node2.up.down = col_node2.down
                col_node2.down.up = col_node2.up
                self.columns[col_node2.col].size -= 1
                col_node2 = col_node2.right
            row_node = row_node.down

    def uncover(self, col_node):
        """取消覆盖"""
        # print("DLX.uncover 被 %s 线程调用了!" % threading.current_thread().name)
        row_node = col_node.up
        while row_node != col_node:
            col_node2 = row_node.left
            while col_node2 != row_node:
                self.columns[col_node2.col].size += 1
                col_node2.up.down = col_node2
                col_node2.down.up = col_node2
                col_node2 = col_node2.left
            row_node = row_node.up

        col_node.right.left = col_node
        col_node.left.right = col_node

    def solve(self, solution):
        """递归求解"""
        # print("DLX.solve 被 %s 线程调用了!" % threading.current_thread().name)

        def _solve(self, solution):
            """递归求解"""
            # print("DLX.solve._solve 被调用了!")
            self.steps += 1
            # print(f"DLX.solve._solve 步数: {self.steps}")
            if self.steps % 10000 == 0:  # 每10000步打印一次
                print(f"进度: {self.steps}步 | 当前解长度: {len(solution)}")
            gc.collect()

            if self.header.right == self.header:
                return True  # 找到解

            # 选择最小的列
            col_node = self.header.right
            min_size = col_node.size
            current = col_node.right
            while current != self.header:
                if current.size < min_size:
                    col_node = current
                    min_size = current.size
                current = current.right

            if col_node.size == 0:
                return False  # 无解

            self.cover(col_node)

            row_node = col_node.down
            while row_node != col_node:
                solution.append(row_node)

                # 覆盖该行中的所有列
                col_node2 = row_node.right
                while col_node2 != row_node:
                    self.cover(self.columns[col_node2.col])
                    col_node2 = col_node2.right

                if self.solve(solution):
                    return True

                # 回溯
                solution.pop()
                col_node2 = row_node.left
                while col_node2 != row_node:
                    self.uncover(self.columns[col_node2.col])
                    col_node2 = col_node2.left

                row_node = row_node.down

            self.uncover(col_node)
            return False
        return _solve(self=self, solution=solution)


print("State: 5/5 - 定义DLX类完成")


def main():
    print("State: Final Stage - 开始求解")
    # 计算所有放置和矩阵行
    global pieces, GRID_WIDTH, GRID_HEIGHT
    matrix_rows, piece_info = build_matrix_and_placements(pieces, GRID_WIDTH, GRID_HEIGHT)

    # 计算总列数
    num_cell_cols = GRID_WIDTH * GRID_HEIGHT
    num_piece_cols = sum(piece.count for piece in pieces)
    total_cols = num_cell_cols + num_piece_cols
    print(f"State: Final Stage - 总列数: {total_cols}")

    # 初始化DLX
    dlx = DLX(total_cols)
    print("State: Final Stage - DLX初始化完成")

    # 添加所有行到DLX，并绑定 piece_info 索引
    for idx, row_columns in enumerate(matrix_rows):
        dlx.add_row(row_columns, row_info_idx=idx)
    print("State: Final Stage - 矩阵行添加完成")

    print(f"Header状态: {dlx.header.right == dlx.header}")
    print(f"列数: {len([col for col in dlx.columns if col.size > 0])}")

    # 检查第一列
    first_col = dlx.header.right
    if first_col != dlx.header:
        print(f"第一列 {first_col.col} 大小: {first_col.size}")
    else:
        print("没有有效的列！")

    # 求解
    solution = []
    if dlx.solve(solution):
        print("找到解！")
        # 解析解
        grid = [[0] * GRID_HEIGHT for _ in range(GRID_WIDTH)]
        for node in solution:
            # 获取 piece_info 索引
            row_idx = getattr(node, 'piece_info_idx', None)
            if row_idx is None:
                cur = node
                while getattr(cur, 'piece_info_idx', None) is None:
                    cur = cur.right
                row_idx = cur.piece_info_idx
            piece_idx, copy_idx, x_start, y_base, is_rotated = piece_info[row_idx]
            # 选择原始或旋转后的piece
            if is_rotated:
                piece = PuzzlePiece([(-dy, length) for dy, length in reversed(pieces[piece_idx].segments)], pieces[piece_idx].count)
            else:
                piece = pieces[piece_idx]
            covered_cells = get_covered_cells(piece, x_start, y_base)
            for x, y in covered_cells:
                grid[x][y] = piece_idx + 1
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                print(f"{grid[x][y]:2d}", end=" ")
            print()
    else:
        print("无解")


def pre_run_check():
    total_placements = 0
    for piece in pieces:
        placements = len(list(get_valid_placements(piece)))
        total_placements += placements * piece.count
        print(f"块 {piece.segments}: {placements}种放置 × {piece.count}")

    print(f"总放置选择: {total_placements}")
    print(f"预计内存: {total_placements * 8 * 50 / 1024 / 1024:.1f} MB")


def monitor_resources():
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    print(f"内存使用: {mem.rss / 1024 / 1024:.1f} MB (RSS)")
    print(f"虚拟内存: {mem.vms / 1024 / 1024:.1f} MB")
    print(f"CPU使用: {process.cpu_percent()}%")
    print(f"线程数: {process.num_threads()}")


# 监控线程函数
def monitor_thread(main_thread, interval=60):
    print("监控线程: 启动监控...")
    segs = 120
    for _ in range(10 * segs):
        time.sleep(10 / segs)
    # 防止监控报告和初始化日志跑一起去
    while main_thread.is_alive():

        # 检查主线程状态
        print("\n--- 监控报告 ---")
        print(f"主线程状态: {'活跃' if main_thread.is_alive() else '终止'}")

        # 监控系统资源
        monitor_resources()

        # 间隔时间
        # 分段睡眠（每次检查exit_event）
        for _ in range(interval * segs):
            time.sleep(interval / segs)
        time.sleep(0.1)

    print("\n监控线程: 检测到主线程已完成，监控结束")


def get_model(model_id):
    with open("models.json", "r", encoding="utf-8") as f:
        import json
        models = json.load(f)
    model_id_str = str(model_id)
    if model_id_str in models:
        model = models[model_id_str]
        pieces = []
        for piece_data in model["pieces"]:
            segments = [(seg[0], seg[1]) for seg in piece_data["segments"]]
            count = piece_data["count"]
            pieces.append(PuzzlePiece(segments, count))
        return pieces, model["grid_width"], model["grid_height"]
    return None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLXSolver - 拼图解算器")
    parser.add_argument("--interval", type=int, default=10, help="监控间隔时间（秒）")
    parser.add_argument("--test", action="store_true", help="启用测试模式（1x1拼图）")
    parser.add_argument("--pieces", type=list, help="自定义拼图块配置")
    parser.add_argument("--grid-width", type=int, help="拼图网格宽度")
    parser.add_argument("--grid-height", type=int, help="拼图网格高度")
    parser.add_argument("--model-id", type=int, default=1593241,
                        help="模型ID, 若代码中包含此模型ID，则将使用此模型的拼图块和网格尺寸")
    # model-id 不能与 pieces, grid-width, grid-height 一起使用

    args = parser.parse_args()
    if args.model_id is not None:
        if args.pieces or args.grid_width or args.grid_height:
            print("错误: model-id 与 pieces, grid-width, grid-height 不能一起使用")
            exit(1)
        else:
            pieces, GRID_WIDTH, GRID_HEIGHT = get_model(args.model_id)
    else:
        if not args.pieces or not args.grid_width or not args.grid_height:
            print("错误: model-id 未指定，且 pieces, grid-width, grid-height 必须全部指定")
            exit(1)
        pieces, GRID_WIDTH, GRID_HEIGHT = args.pieces, args.grid_width, args.grid_height
    if args.test:
        pieces, GRID_WIDTH, GRID_HEIGHT = create_test_case()
        print("测试模式启用: 1x1拼图")
    pre_run_check()
    main_th = threading.Thread(target=main)
    main_th.start()
    monitor_th = threading.Thread(
        target=monitor_thread, args=(main_th, args.interval))
    monitor_th.daemon = True
    monitor_th.start()
    main_th.join()
    monitor_th.join()
