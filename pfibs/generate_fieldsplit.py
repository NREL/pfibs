from dolfin import *
# block = {
#     0: {
#         0: {
#             0: {
#                 "function_space": [[0,0]],
#                 "solve": "preonly",
#                 "pc": "hypre"
#                 "hypre_threshold": 0.5
#                 },
#             1: [[[1,0]],"preonly","hypre"]
#             "fieldsplit_type": "schur"
#             "schur_type": "selfp"
#         },
#         1: [[[2,0]], "preonly", "hypre"],
#         "fieldsplit_type": "additive"
#     }
#     1: {
#         0: {
#             0: [[[0,1]],"preonly","hypre"],
#             1: [[[1,1]],"preonly","hypre"],
#             "fieldsplit_type": "schur",
#             "schur_type": "selfp"
#         },
#         1: [[[2,1]], "preonly", "hypre"]
#         "fieldsplit_type": "additive"
#     }
# }


block = [
[(0,1), [2]],
[[3,4], [5]],
[[6], (7,8)],
[[11,9], [10]]
]

# block = [
# [(0,1), [2]],
# [[3,4], [5]]
# ]

prefix = ""



def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    if isinstance(lis,int):
        new_lis.append(lis)
    else:
        for item in lis:
            new_lis.extend(flatten(item))
    return new_lis

def shift(lis,n=0):
    """Returns [lis, n]. lis is the input lis with the same nesting but numbered 0-n. n is the final index"""
    if isinstance(lis,int):
        return [n,n]
    else:
        if isinstance(lis,list):
            n-=1
            for i in range(len(lis)):
                n+=1
                out = shift(lis[i],n)
                lis[i] = out[0]
                n = out[1]
        elif isinstance(lis,tuple):
            lis = list(lis)
            n-=1
            for i in range(len(lis)):
                n+=1
                out = shift(lis[i],n)
                lis[i] = out[0]
                n = out[1]
            lis = tuple(lis)
        return [lis,n]

def build_fieldsplit(block,prefix):
    n = len(block)
    if n !=1:
        # PETScOptions.set(prefix+"pc_type","fieldsplit")
        print("PETScOptions.set('"+prefix+"pc_type"+"', '"+"fieldsplit"+"')")

    for i in range(n):

        if isinstance(block[i],int):
            pass

        else:
            newprefix = "fieldsplit_"+repr(i)+"_"
            flat = flatten(block[i])
            flat_str = str(flat).strip("[]")

            # PETScOptions.set(prefix+"pc_"+newprefix+"fields",flat)
            print("PETScOptions.set('"+prefix+"pc_"+newprefix+"fields"+"', '"+flat_str+"')")

            if isinstance(block[i],tuple):
                print("PETScOptions.set('"+prefix+newprefix+"fieldsplit_type"+"', '"+"schur"+"')")


            build_fieldsplit(shift(block[i])[0],prefix+newprefix)

build_fieldsplit(block,prefix)





# PETScOptions.set('pc_type', 'fieldsplit')
# PETScOptions.set('pc_fieldsplit_0_fields', '0, 1, 2')
# PETScOptions.set('fieldsplit_0_pc_type', 'fieldsplit')

# PETScOptions.set('fieldsplit_0_pc_fieldsplit_0_fields', '0, 1')
# PETScOptions.set('fieldsplit_0_fieldsplit_0_fieldsplit_type', 'schur')
# PETScOptions.set('fieldsplit_0_fieldsplit_0_pc_type', 'fieldsplit')

# PETScOptions.set('fieldsplit_0_pc_fieldsplit_1_fields', '2')



# PETScOptions.set('pc_fieldsplit_1_fields', '3, 4, 5')
# PETScOptions.set('fieldsplit_1_pc_type', 'fieldsplit')

# PETScOptions.set('fieldsplit_1_pc_fieldsplit_0_fields', '0, 1')
# PETScOptions.set('fieldsplit_1_fieldsplit_0_pc_type', 'fieldsplit')

# PETScOptions.set('fieldsplit_1_pc_fieldsplit_1_fields', '2')



# PETScOptions.set('pc_fieldsplit_2_fields', '6, 7, 8')
# PETScOptions.set('fieldsplit_2_pc_type', 'fieldsplit')

# PETScOptions.set('fieldsplit_2_pc_fieldsplit_0_fields', '0')

# PETScOptions.set('fieldsplit_2_pc_fieldsplit_1_fields', '1, 2')
# PETScOptions.set('fieldsplit_2_fieldsplit_1_fieldsplit_type', 'schur')
# PETScOptions.set('fieldsplit_2_fieldsplit_1_pc_type', 'fieldsplit')



# PETScOptions.set('pc_fieldsplit_3_fields', '11, 9, 10')
# PETScOptions.set('fieldsplit_3_pc_type', 'fieldsplit')

# PETScOptions.set('fieldsplit_3_pc_fieldsplit_0_fields', '0, 1')
# PETScOptions.set('fieldsplit_3_fieldsplit_0_pc_type', 'fieldsplit')

# PETScOptions.set('fieldsplit_3_pc_fieldsplit_1_fields', '2')