#include<bits/stdc++.h>
using namespace std;

vector<int> bfs(vector<vector<int>>& adj)  {
    int V = adj.size();
    
    int s = 0; // source node 
    vector<int> res;

  
    queue<int> q;  
    
  
    vector<bool> visited(V, false);


    visited[s] = true;
    q.push(s);

    while (!q.empty()) {
      
        int curr = q.front();
        q.pop();
        res.push_back(curr);

        for (int x : adj[curr]) {
            if (!visited[x]) {
                visited[x] = true;
                q.push(x);
            }
        }
    }

    return res;
}

int main()  {

    vector<vector<int>> adj = {{1,2}, {0,2,3}, {0,4}, {1,4}, {2,3}};
    vector<int> ans = bfs(adj);
    for(auto i:ans) {
        cout<<i<<" ";
    }
    return 0;
}