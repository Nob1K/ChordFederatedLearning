struct node {
    1: string ip,
    2: i32 port,
    3: i32 id
}

struct weights {
    1: list<list<double>> w,
    2: list<list<double>> v,
    3: i32 status // 0 for normal, 1 for wait, -1 for error
}

service compute {
    oneway void put_data(1:string filename),
    weights get_model(1:string filename),
    void print_info(),
    node find_successor(1: i32 id),
    void notify(1: node new_node),
    node get_predecessor(),
    // stabilization
    list<node> get_successor_list(),
    void ping(),
    // send a trained model to a backup node
    oneway void replicate_model(1:string filename, 2:weights w)
}
