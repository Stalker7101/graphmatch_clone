import os
import javalang
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node
from anytree import AnyNode


edges = {
    "Nexttoken": 2,
    "Prevtoken": 3,
    "Nextuse": 4,
    "Prevuse": 5,
    "If": 6,
    "Ifelse": 7,
    "While": 8,
    "For": 9,
    "Nextstmt": 10,
    "Prevstmt": 11,
    "Prevsib": 12,
}


def get_edge_next_sib(node, vocab_dict, src, tgt, edge_type):
    for i in range(len(node.children) - 1):
        src.append(node.children[i].id)
        tgt.append(node.children[i + 1].id)
        edge_type.append([1])
        src.append(node.children[i + 1].id)
        tgt.append(node.children[i].id)
        edge_type.append([edges["Prevsib"]])
    for child in node.children:
        get_edge_next_sib(child, vocab_dict, src, tgt, edge_type)


def edge_flow_append(edge, node, src, tgt, edge_type):
    src.append(node.children[0].id)
    tgt.append(node.children[1].id)
    edge_type.append([edges[edge]])
    src.append(node.children[1].id)
    tgt.append(node.children[0].id)
    edge_type.append([edges[edge]])


def get_edge_flow(
    node,
    vocab_dict,
    src,
    tgt,
    edge_type,
    if_edge=False,
    while_edge=False,
    for_edge=False,
):
    token = node.token
    if while_edge == True and token == "WhileStatement":
        edge_flow_append("While", node, src, tgt, edge_type)
    if for_edge == True and token == "ForStatement":
        edge_flow_append("For", node, src, tgt, edge_type)
    if if_edge == True and token == "IfStatement":
        edge_flow_append("If", node, src, tgt, edge_type)
        if len(node.children) == 3:
            src.append(node.children[0].id)
            tgt.append(node.children[2].id)
            edge_type.append([edges["Ifelse"]])
            src.append(node.children[2].id)
            tgt.append(node.children[0].id)
            edge_type.append([edges["Ifelse"]])
    for child in node.children:
        get_edge_flow(
            child, vocab_dict, src, tgt, edge_type, if_edge, while_edge, for_edge
        )


def get_edge_next_stmt(node, vocab_dict, src, tgt, edge_type):
    token = node.token
    if token == "BlockStatement":
        for i in range(len(node.children) - 1):
            src.append(node.children[i].id)
            tgt.append(node.children[i + 1].id)
            edge_type.append([edges["Nextstmt"]])
            src.append(node.children[i + 1].id)
            tgt.append(node.children[i].id)
            edge_type.append([edges["Prevstmt"]])
    for child in node.children:
        get_edge_next_stmt(child, vocab_dict, src, tgt, edge_type)


def get_edge_next_token(node, vocab_dict, src, tgt, edge_type, token_list):
    def get_token_list(node, vocab_dict, edge_type, token_list):
        if len(node.children) == 0:
            token_list.append(node.id)
        for child in node.children:
            get_token_list(child, vocab_dict, edge_type, token_list)

    get_token_list(node, vocab_dict, edge_type, token_list)
    for i in range(len(token_list) - 1):
        src.append(token_list[i])
        tgt.append(token_list[i + 1])
        edge_type.append([edges["Nexttoken"]])
        src.append(token_list[i + 1])
        tgt.append(token_list[i])
        edge_type.append([edges["Prevtoken"]])


def get_edge_next_use(node, vocab_dict, src, tgt, edge_type, variable_dict):
    def get_variables(node, vocab_dict, edge_type, variable_dict):
        token = node.token
        if token == "MemberReference":
            for child in node.children:
                if child.token == node.data.member:
                    variable = child.token
                    variable_node = child
            if not variable_dict.__contains__(variable):
                variable_dict[variable] = [variable_node.id]
            else:
                variable_dict[variable].append(variable_node.id)
        for child in node.children:
            get_variables(child, vocab_dict, edge_type, variable_dict)

    get_variables(node, vocab_dict, edge_type, variable_dict)
    for v in variable_dict.keys():
        for i in range(len(variable_dict[v]) - 1):
            src.append(variable_dict[v][i])
            tgt.append(variable_dict[v][i + 1])
            edge_type.append([edges["Nextuse"]])
            src.append(variable_dict[v][i + 1])
            tgt.append(variable_dict[v][i])
            edge_type.append([edges["Prevuse"]])


def get_token(node):
    token = ""
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = "Modifier"
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def get_child(root):
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif item:
                yield item

    return list(expand(children))


def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    for child in children:
        get_sequence(child, sequence)


def get_nodes(node, node_list):
    node_list.append(node)
    children = get_child(node)
    for child in children:
        get_nodes(child, node_list)


def create_tree(root, node, node_list, parent=None):
    id = len(node_list)
    token, children = get_token(node), get_child(node)
    if id == 0:
        root.token = token
        root.data = node
    else:
        new_node = AnyNode(id=id, token=token, data=node, parent=parent)
    node_list.append(node)
    for child in children:
        if id == 0:
            create_tree(root, child, node_list, parent=root)
        else:
            create_tree(root, child, node_list, parent=new_node)


def get_node_and_edge_ast_only(node, node_index_list, vocab_dict, src, tgt):
    token = node.token
    node_index_list.append([vocab_dict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        get_node_and_edge_ast_only(child, node_index_list, vocab_dict, src, tgt)


def get_node_and_edge(node, node_index_list, vocab_dict, src, tgt, edge_type):
    token = node.token
    node_index_list.append([vocab_dict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edge_type.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edge_type.append([0])
        get_node_and_edge(child, node_index_list, vocab_dict, src, tgt, edge_type)


def count_nodes(node, if_count, while_count, for_count, block_count):
    token = node.token
    if token == "IfStatement":
        if_count += 1
    if token == "WhileStatement":
        while_count += 1
    if token == "ForStatement":
        for_count += 1
    if token == "BlockStatement":
        block_count += 1
    print(if_count, while_count, for_count, block_count)
    for child in node.children:
        count_nodes(child, if_count, while_count, for_count, block_count)


def create_ast():
    asts, paths, all_tokens = [], [], []
    dir_name = "BCB/bigclonebenchdata/"
    for rt, _dirs, files in os.walk(dir_name):
        for file in files:
            program_file = open(os.path.join(rt, file), encoding="utf-8")
            program_text = program_file.read()
            program_tokens = javalang.tokenizer.tokenize(program_text)
            parser = javalang.parse.Parser(program_tokens)
            program_ast = parser.parse_member_declaration()
            paths.append(os.path.join(rt, file))
            asts.append(program_ast)
            get_sequence(program_ast, all_tokens)
            program_file.close()
    ast_dict = dict(zip(paths, asts))
    if_count, while_count, for_count, block_count, do_count, switch_count = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    for token in all_tokens:
        if token == "IfStatement":
            if_count += 1
        if token == "WhileStatement":
            while_count += 1
        if token == "ForStatement":
            for_count += 1
        if token == "BlockStatement":
            block_count += 1
        if token == "DoStatement":
            do_count += 1
        if token == "SwitchStatement":
            switch_count += 1
    print(if_count, while_count, for_count, block_count, do_count, switch_count)
    print("all_nodes ", len(all_tokens))
    all_tokens = list(set(all_tokens))
    vocab_size = len(all_tokens)
    token_ids = range(vocab_size)
    vocab_dict = dict(zip(all_tokens, token_ids))
    print(vocab_size)
    return ast_dict, vocab_size, vocab_dict


def create_separate_graph(
    ast_dict,
    vocab_len,
    vocab_dict,
    device,
    mode="ast_only",
    next_sib=False,
    if_edge=False,
    while_edge=False,
    for_edge=False,
    block_edge=False,
    next_token=False,
    next_use=False,
):
    path_list, tree_list = [], []
    print("next_sib ", next_sib)
    print("if_edge ", if_edge)
    print("while_edge ", while_edge)
    print("for_edge ", for_edge)
    print("block_edge ", block_edge)
    print("next_token", next_token)
    print("next_use ", next_use)
    print(len(ast_dict))
    for path, tree in ast_dict.items():
        node_list = []
        new_tree = AnyNode(id=0, token=None, data=None)
        create_tree(new_tree, tree, node_list)
        x, edge_src, edge_tgt, edge_attr = [], [], [], []
        if mode == "ast_only":
            get_node_and_edge_ast_only(new_tree, x, vocab_dict, edge_src, edge_tgt)
        else:
            get_node_and_edge(new_tree, x, vocab_dict, edge_src, edge_tgt, edge_attr)
            if next_sib == True:
                get_edge_next_sib(new_tree, vocab_dict, edge_src, edge_tgt, edge_attr)
            get_edge_flow(
                new_tree,
                vocab_dict,
                edge_src,
                edge_tgt,
                edge_attr,
                if_edge,
                while_edge,
                for_edge,
            )
            if block_edge == True:
                get_edge_next_stmt(new_tree, vocab_dict, edge_src, edge_tgt, edge_attr)
            tokenlist = []
            if next_token == True:
                get_edge_next_token(
                    new_tree, vocab_dict, edge_src, edge_tgt, edge_attr, tokenlist
                )
            variable_dict = {}
            if next_use == True:
                get_edge_next_use(
                    new_tree, vocab_dict, edge_src, edge_tgt, edge_attr, variable_dict
                )
        edge_index = [edge_src, edge_tgt]
        ast_length = len(x)
        path_list.append(path)
        tree_list.append([[x, edge_index, edge_attr], ast_length])
        ast_dict[path] = [[x, edge_index, edge_attr], ast_length]
    return ast_dict


def create_gmn_data(id, tree_dict, vocab_len, vocab_dict, device):
    index_dir = "BCB/"
    if id == "0" or id == "11":
        train_file, valid_file, test_file = (
            open(index_dir + ("traindata.txt" if id == "0" else "traindata11.txt")),
            open(index_dir + "devdata.txt"),
            open(index_dir + "testdata.txt"),
        )
    else:
        print("file not exist")
        quit()
    train_list, valid_list, test_list = (
        train_file.readlines(),
        valid_file.readlines(),
        test_file.readlines(),
    )
    train_data, valid_data, test_data = [], [], []
    print("train_data")
    train_data = create_pair_data(tree_dict, train_list, device=device)
    print("valid_data")
    valid_data = create_pair_data(tree_dict, valid_list, device=device)
    print("test_data")
    test_data = create_pair_data(tree_dict, test_list, device=device)
    return train_data, valid_data, test_data


def create_pair_data(tree_dict, path_list, device):
    data_list = []
    count_lines = 1
    for line in path_list:
        count_lines += 1
        pair_info = line.split()
        code_1_path = "BCB" + pair_info[0].strip(".")
        code_2_path = "BCB" + pair_info[1].strip(".")
        label = int(pair_info[2])
        data_1 = tree_dict[code_1_path]
        data_2 = tree_dict[code_2_path]
        x_1, edge_index_1, edge_attr_1, ast_1_length = (
            data_1[0][0],
            data_1[0][1],
            data_1[0][2],
            data_1[1],
        )
        x_2, edge_index_2, edge_attr_2, ast_2_length = (
            data_2[0][0],
            data_2[0][1],
            data_2[0][2],
            data_2[1],
        )
        if edge_attr_1 == []:
            edge_attr_1 = None
            edge_attr_2 = None
        data = [[x_1, x_2, edge_index_1, edge_index_2, edge_attr_1, edge_attr_2], label]
        data_list.append(data)
    return data_list


if __name__ == "__main__":
    ast_dict, vocab_size, vocab_dict = create_ast()
    treedict = create_separate_graph(
        ast_dict,
        vocab_size,
        vocab_dict,
        device="gpu",
        mode="else",
        next_sib=True,
        if_edge=True,
        while_edge=True,
        for_edge=True,
        block_edge=True,
        next_token=True,
        next_use=True,
    )
