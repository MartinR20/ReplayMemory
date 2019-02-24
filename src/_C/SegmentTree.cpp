#include <iostream>

template<typename T>
class AppendableSegmentTree {
  private:
    unsigned int n;
    unsigned int idx;
    T* tree; 

  public:
    // function to build the tree 
    AppendableSegmentTree(const unsigned int n) : n(n), idx(0), tree(new T[2*n]) {
      for(unsigned int i = 0; i < 2*n; ++i) {
        tree[i] = 0;
      }      
    }

    // function to update a tree node 
    void update(unsigned int p, T value)  
    {  
      // set value at position p 
      tree[p+n] = value; 
      p = p+n; 
        
      // move upward and update parents 
      for (unsigned int i=p; i > 1; i >>= 1)
        tree[i>>1] = tree[i] + tree[i^1];
    } 
    
    void append(T value)
    {
      update(idx, value);
      ++idx;
    }

    // function to get sum on interval [l, r) 
    T query(unsigned int l, unsigned int r)  
    {  
      T res = 0; 
        
      // loop to find the sum in the range 
      for (l += n, r += n; l < r; l >>= 1, r >>= 1) 
      { 
        if (l&1)  
          res += tree[l++]; 
      
        if (r&1)  
          res += tree[--r]; 
      } 
        
      return res; 
    } 

    T find(T value) {
      unsigned int i = 1;

      for (; i < (2*n); i<<=1) {
        if(value > tree[i]) {
          value -= tree[i];
          ++i;
        }
      }

      return tree[i>>1]; 
    }
    
    T total() {
      return tree[1];
    }
    
    bool full() {
      return idx >= n;
    }

    void print() {
      for(unsigned int i = 0; i < 2*n; ++i) {
        for(unsigned int padd = 0; padd < (2 * n / (2* (i + 1))); ++padd)
          std::cout << ' ';
        std::cout << tree[i] << (((i  + 1) & i) == 0 ? '\n' : ' ');
      }
      std::cout << '\n';
    } 
};

/*
int main()  
{ 
  float a[] = {42, 8, 16, 43, 4, 40, 78, 86, 50, 1, 5, 56, 7, 32, 10, 98, 92, 62, 28, 52,
               97, 87, 22, 29, 27, 85, 11, 70, 34, 6, 58, 37, 47, 51, 35, 48, 46, 61, 44,
               14, 65, 63, 45, 19, 21, 88, 31, 24, 94, 75, 67, 3, 2, 99, 30, 95, 38, 12, 
               49, 23, 39, 71, 33, 55, 13, 84, 89, 64, 54, 59, 20, 18, 25, 26, 60, 53, 80, 
               82, 9, 77, 69, 76, 57, 91, 68, 74, 66, 15, 72, 41, 73, 100, 17, 96, 79, 83, 
               93, 36, 90, 81};

  // build tree  
  AppendableSegmentTree<float> segtree = AppendableSegmentTree<float>(100); 
   
  for(unsigned int i = 0; i < 50; i++) segtree.append(a[i]); 

  segtree.print();

  std::cout << segtree.total() << '\n';

  return 0; 
} 
*/