import numpy as np
import copy


# problem_path = 'C:/Users/Fragment/Desktop/benchmark/problems_30.npy'
# problem = np.load(problem_path, allow_pickle=True)
#
# problem_idx = 0
# adj_mat = problem[problem_idx][0]
# resource_information = problem[problem_idx][1]
# resource_capacity = problem[problem_idx][2]
# test_adj = copy.deepcopy(adj_mat)
# test_nodes = copy.deepcopy(nodes_information)
# test_resource = copy.deepcopy(resource)
#
# resource_variant = []
#
# agent = A2C()
# env = Environment()
# state = env.reset(test_adj, test_nodes, test_resource, resource_variant, agent)
# job_dag = env.executor.jobdag_map
# msg_mat, msg_mask = get_bottom_up_paths(env.executor.jobdag_map, max_depth=256)


# problem_path = './PSPLIB_dataset/problems_30.npy'
# problem = np.load(problem_path, allow_pickle=True)
#
# problem_idx = 0
# adj_mat = problem[problem_idx][0]
# resource_information = problem[problem_idx][1]
# resource_capacity = problem[problem_idx][2]
# test_adj = copy.deepcopy(adj_mat)
# test_nodes = copy.deepcopy(resource_information)
# test_resource = copy.deepcopy(resource_capacity)
#
# resource_variant = []
#############################
# test_adj = adj_mat
# test_nodes = nodes_information11
# test_resource = resource
# resource_variant = []
#
# env = Environment()
# agent = A2C()
#
# state = env.reset(test_adj, test_nodes, test_resource, resource_variant, agent)
#
# job_dag = env.executor.jobdag_map
#
# max_depth = 7

class SparseMat(object):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.row = []
        self.col = []
        self.data = []

    def add(self, row, col, data):
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def get_col(self):
        return np.array(self.col)

    def get_row(self):
        return np.array(self.row)

    def get_data(self):
        return np.array(self.data)


def get_init_frontier(job_dag, depth):
    sources = set(job_dag.nodes)
    for d in range(depth):
        new_sources = set()
        for n in sources:
            # print(n)
            if len(n.child_nodes) == 0:
                new_sources.add(n)
            else:
                tep_set = set()
                for i in n.child_nodes:
                    tep_set.add(job_dag.nodes[i])
                new_sources.update(tep_set)

        sources = new_sources

    frontier = sources
    return frontier

############调试完成###############


def get_bottom_up_paths(job_dag, max_depth):
    """
    The paths start from all leaves and end with
    frontier (parents all finished) unfinished nodes
    """
    num_nodes = job_dag.num_nodes

    msg_mats = []
    msg_masks = np.zeros([max_depth, num_nodes])

    # get set of frontier nodes in the beginning
    # this is constrained by the message passing depth
    frontier = get_init_frontier(job_dag, max_depth)
    msg_level = {}

    # initial nodes are all message passed
    for n in frontier:
        msg_level[n] = 0

    # pass messages
    for depth in range(max_depth):
        new_frontier = set()
        parent_visited = set()  # save some computation
        for n in frontier:
# 原文parent_nodes是对象，但我的程序中是索引，所以需要通过索引找到对象
            # 通过索引找到对象
            n_parent_nodes_duixiang = []
            for idx in n.parent_nodes:
                n_parent_nodes_duixiang.append(job_dag.nodes[idx])
######修改分界线######
            for parent in n_parent_nodes_duixiang:
                if parent not in parent_visited:
                    curr_level = 0
                    children_all_in_frontier = True

                    # 同上，parent.child_nodes是索引，要找到它的对象
                    duixiang2 = []
                    for idx in parent.child_nodes:
                        duixiang2.append(job_dag.nodes[idx])
                    ############
                    for child in duixiang2:
                        if child not in frontier:
                            children_all_in_frontier = False
                            break
                        if msg_level[child] > curr_level:
                            curr_level = msg_level[child]
                    # children all ready
                    if children_all_in_frontier:
                        if parent not in msg_level or \
                           curr_level + 1 > msg_level[parent]:
                            # parent node has deeper message passed
                            new_frontier.add(parent)
                            msg_level[parent] = curr_level + 1
                    # mark parent as visited
                    parent_visited.add(parent)

        if len(new_frontier) == 0:
            break  # some graph is shallow

        # assign parent-child path in current iteration
        sp_mat = SparseMat(dtype=np.float32, shape=(num_nodes, num_nodes))
        for n in new_frontier:
            # 这里还是对象问题
            duixiang3 = []
            for idx in n.child_nodes:
                duixiang3.append(job_dag.nodes[idx])
            ######
            for child in duixiang3:
                sp_mat.add(row=n.idx, col=child.idx, data=1)
            msg_masks[depth, n.idx] = 1
        msg_mats.append(sp_mat)

        # Note: there might be residual nodes that
        # can directly pass message to its parents
        # it needs two message passing steps
        # (e.g., TPCH-17, node 0, 2, 4)
        for n in frontier:
            parents_all_in_frontier = True
            for p in n.parent_nodes:
                if not p in msg_level:
                    parents_all_in_frontier = False
                    break
            if not parents_all_in_frontier:
                new_frontier.add(n)

        # start from new frontier
        frontier = new_frontier

    # deliberately make dimension the same, for batch processing
    for _ in range(depth, max_depth):
        msg_mats.append(SparseMat(dtype=np.float32,
            shape=(num_nodes, num_nodes)))

    return msg_mats, msg_masks
########################这个函数至少能正常运行################


def generate_CPM(job_dag, msg_mat, msg_mask):
    CPM_time_list = np.zeros(job_dag.num_nodes)
    CPM_time_list[len(msg_mask[0]) - 1] = job_dag.nodes[len(msg_mask[0])- 1].rest_time
    for mat in msg_mat:
        if len(mat.col) == 0:
            break
        col = mat.col
        row = mat.row
        # row_col = np.vstack([row, col])
        row_set = set(row)
        for child_idx in row_set:
            same_time = row.count(child_idx)
            id1 = [i for i, x in enumerate(row) if x == child_idx]
            max_time = 0
            for i in id1:
                # if CPM_time_list[col[i]] != 0:
                if CPM_time_list[col[i]] > max_time:
                    max_time = CPM_time_list[col[i]]
                # else:
                #     if job_dag.nodes[col[i]].rest_time > max_time:
                #        max_time = job_dag.nodes[col[i]].rest_time
                #        print(1)
            CPM_time_list[child_idx] = max_time + job_dag.nodes[child_idx].rest_time
    # print(CPM_time_list)
    return CPM_time_list

##############################################
def CPM_agent(state, CPM_time_list):
    choose_idx = state.runable_nodes_idx
    max_CPM = 0
    action = None
    for i in choose_idx:
        if CPM_time_list[i] >= max_CPM:
            max_CPM = CPM_time_list[i]
            action = i
    return action

def random_agent(state, CPM_time_list):
    choose_idx = state.runable_nodes_idx
    action = np.random.choice(choose_idx, 1)
    action = action[0]
    return action

def SPT_agent(state):
    choose_idx = state.runable_nodes_idx