#!/usr/bin/env python
# coding: utf-8

# In[1]:


#代码示例：4-1
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
X=tf.constant(
    [
        [
            [[3],[-1],[3]],
            [[9],[2],[4]],
            [[8],[1],[5]]
        ]
    ]
    ,tf.float32
)
K=tf.constant(
    [
        [
            [[5]],[[2]]],
        [
            [[1]],[[3]]
        ]
    ]
    ,tf.float32
)
# same卷积
conv=tf.nn.conv2d(X,K,(1,1,1,1),'SAME')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[2]:


#代码示例：4-2
import tensorflow as tf
X=tf.constant(
    [
        [
            [[3],[-1],[3]],
            [[9],[2],[4]],
            [[8],[1],[5]]
        ]
    ]
    ,tf.float32
)
K=tf.constant(
    [
        [
            [[5]],[[2]]],
        [
            [[1]],[[3]]
        ]
    ]
    ,tf.float32
)
# valid卷积
conv=tf.nn.conv2d(X,K,(1,1,1,1),'VALID')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[3]:


#代码示例：4-3
import tensorflow as tf
X=tf.constant(
    [
        [
            [[3,1],[-1,4],[3,2]],
            [[9,7],[2,-1],[4,3]],
            [[8,-2,],[1,5],[5,9]]
        ]
    ]
    ,tf.float32
)
K=tf.constant(
    [
        [
            [[5],[2]],[[2],[1]]],
        [
            [[1],[6]],[[3],[-4]]
        ]
    ]
    ,tf.float32
)
# valid卷积
conv=tf.nn.conv2d(X,K,(1,1,1,1),'VALID')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[4]:


#代码示例：4-4
import tensorflow as tf
X=tf.constant(
    [
        [
            [[3,1],[-1,4],[3,2]],
            [[9,7],[2,-1],[4,3]],
            [[8,-2,],[1,5],[5,9]]
        ]
    ]
    ,tf.float32
)
K=tf.constant(
    [
        [
            [[5,-1,2],[2,-2,1]],[[2,2,-7],[1,5,1]]],
        [
            [[1,3,8],[6,8,2]],[[3,4,3],[-4,3,-2]]
        ]
    ]
    ,tf.float32
)
# valid卷积
conv=tf.nn.conv2d(X,K,(1,1,1,1),'VALID')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[5]:


#代码示例：4-5
import tensorflow as tf

X=tf.constant(
    [
        [
            [[3,1],[-1,4],[3,2]],
            [[9,7],[2,-1],[4,3]],
            [[8,-2,],[1,5],[5,9]]
        ],
         [
            [[1,2],[1,2],[2,-2]],
            [[-3,4],[2,-3],[2,1]],
            [[5,-1],[3,1],[5,3]]
         ]
    ]
    ,tf.float32
)

K=tf.constant(
    [
        [
            [[5,-1,2],[2,-2,1]],[[2,2,-7],[1,5,1]]],
        [
            [[1,3,8],[6,8,2]],[[3,4,3],[-4,3,-2]]
        ]
    ]
    ,tf.float32
)
# valid卷积
conv=tf.nn.conv2d(X,K,(1,1,1,1),'VALID')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[6]:


#代码示例：4-6
import tensorflow as tf
X=tf.constant(
    [
        [
            [[3,1],[-1,4],[3,2]],
            [[9,7],[2,-1],[4,3]],
            [[8,-2,],[1,5],[5,9]]
        ]
    ]
    ,tf.float32
)
K=tf.constant(
    [
        [
            [[5],[2]],
            [[2],[1]]],
        [   
            [[1],[6]],
            [[3],[-4]]]
    ]
    ,tf.float32
)
# valid卷积
conv=tf.nn.depthwise_conv2d(X,K,(1,1,1,1),'VALID')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[7]:


#代码示例：4-7
import tensorflow as tf
X=tf.constant(
    [
        [
            [[3,1],[-1,4],[3,2]],
            [[9,7],[2,-1],[4,3]],
            [[8,-2,],[1,5],[5,9]]
        ]
    ]
    ,tf.float32
)
K=tf.constant(
    [
        [
            [[5,-1,2],
             [2,-2,1]],
            [[2,2,-7],
             [1,5,1]]
        ],
        [
            [[1,3,8],
             [6,8,2]],
            [[3,4,3],
             [-4,3,-2]]
        ]
    ]
    ,tf.float32
)
# valid卷积
conv=tf.nn.depthwise_conv2d(X,K,(1,1,1,1),'VALID')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[8]:


#代码示例：4-8
import tensorflow as tf
X=tf.constant(
     [
         [
             [[3],[2],[1],[4]],
             [[8],[1],[5],[9]],
             [[6],[2],[-1],[7]],
             [[-3],[4],[6],[5]]
         ]
     ]
     ,tf.float32)
# same最大值池化
maxPool=tf.nn.max_pool(X,(1,2,3,1),[1,1,1,1],'SAME')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[9]:


#代码示例：4-9
import tensorflow as tf
X=tf.constant(
     [
         [
             [[3,1],[-1,4],[3,2]],
             [[9,7],[2,-1],[4,3]],
             [[8,-2],[1,5],[5,9]]
         ]
     ]
     ,tf.float32)

# 多深度张量的same池化，2行2列2深度的邻域掩码
maxPool=tf.nn.max_pool(X,(1,2,2,1),[1,2,2,1],'SAME')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[10]:


#代码示例：4-10
import tensorflow as tf
X=tf.constant(
     [
        [
             [[2,5],[3,3],[8,2]],
             [[6,1],[1,2],[5,4]],
             [[7,9],[2,-3],[-1,3]]
         ],
         [
             [[3,1],[-1,4],[3,2]],
             [[9,7],[2,-1],[4,3]],
             [[8,-2],[1,5],[5,9]]
         ]
     ]
     ,tf.float32)
# 多个三维张量的same最大值池化，2行2列2深度的邻域掩码
maxPool=tf.nn.max_pool(X,(1,2,2,1),[1,2,2,1],'SAME')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[11]:


#代码示例：4-11
import tensorflow as tf
X=tf.constant(
     [
         [
             [[3,1],[-1,4],[3,2]],
             [[9,7],[2,-1],[4,3]],
             [[8,-2],[1,5],[5,9]]
         ]
     ]
     ,tf.float32)
# same平均值池化，2行2列2深度的邻域掩码
avgPool=tf.nn.avg_pool(X,(1,2,2,1),[1,2,2,1],'SAME')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[12]:


#代码示例4-12
import tensorflow as tf
X=tf.constant(
     [
         [
             [[3],[2],[1],[4]],
             [[8],[1],[5],[9]],
             [[6],[2],[-1],[7]],
             [[-3],[4],[6],[5]]
         ]
     ]
     ,tf.float32)
# valid最大值池化
maxPool=tf.nn.max_pool(X,(1,2,2,1),[1,1,1,1],'VALID')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[13]:


#代码示例4-13
import tensorflow as tf
X=tf.constant(
     [
         [
             [[3,1],[-1,4],[3,2]],
             [[9,7],[2,-1],[4,3]],
             [[8,-2],[1,5],[5,9]]
         ]
     ]
     ,tf.float32)
# 多深度张量的valid池化
maxPool=tf.nn.max_pool(X,(1,2,2,1),[1,1,1,1],'VALID')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[14]:


#代码示例4-14
import tensorflow as tf
X=tf.constant(
     [
         [
             [[3,1],[-1,4],[3,2]],
             [[9,7],[2,-1],[4,3]],
             [[8,-2],[1,5],[5,9]]
         ],
        [
             [[1,4],[9,3],[1,1]],
             [[1,1],[1,2],[3,3]],
             [[2,1],[3,6],[4,2]]
         ]

     ]
     ,tf.float32)
# 多个三维张量的valid池化
maxPool=tf.nn.max_pool(X,(1,2,2,1),[1,1,1,1],'VALID')
session=tf.compat.v1.Session()
print(session.run(conv))


# In[ ]:




