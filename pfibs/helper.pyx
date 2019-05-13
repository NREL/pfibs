import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t
from cpython.mem cimport PyMem_Malloc
def iterate_section(section, full_field_array, num_sub_fields, sub_field_array, total_field_indx, ISarray, sub_field_indx):
    ## Iterate through Section Chart ##
    cdef int pstart, pend, i, i_field, j, field, c_total_field_indx, i_sub_field
    c_total_field_indx =  total_field_indx
    c_full_field_array = <int *>PyMem_Malloc(len(full_field_array)*sizeof(int))
    len_arr = len(full_field_array)
    for i in range(len_arr):
        c_full_field_array[i] = full_field_array[i]
    len_sub_field_array = len(sub_field_array)
    c_sub_field_array = <int **>PyMem_Malloc(len_sub_field_array*sizeof(int*))
    len_sub_field_array_2 = <int *>PyMem_Malloc(len_sub_field_array*sizeof(int))
    for i in range(len_sub_field_array):
        len_sub_field_array_2[i] = len(sub_field_array[i][1])
        c_sub_field_array[i] = <int *>PyMem_Malloc(len_sub_field_array_2[i]*sizeof(int))
        for j in range(len_sub_field_array_2[i]):
            c_sub_field_array[i][j] = sub_field_array[i][1][j]

    len_ISarray = len(ISarray)
    c_ISarray = <int **>PyMem_Malloc(len_ISarray*sizeof(int*))
    for i in range(len_ISarray):
        len_ISarray_2 = len(ISarray[i])
        c_ISarray[i] = <int *>PyMem_Malloc(len_ISarray_2*sizeof(int))
        for j in range(len_ISarray_2):
            c_ISarray[i][j] = ISarray[i][j]


    c_sub_field_indx = <int *>PyMem_Malloc(len(sub_field_indx)*sizeof(int))
    for i in range(len(sub_field_indx)):
        c_sub_field_indx[i] = sub_field_indx[i]

    cdef int c_num_sub_fields = num_sub_fields
    cdef int numDof
    pstart,pend = section.getChart()
    for i in range(pstart,pend):
        for i_field in range(len_arr):
            field = c_full_field_array[i_field]
            ## Check if this DoF is associated with the given field ##
            numDof = section.getFieldDof(i,field)

            ## If it is, find which sub field array it belongs to ##
            if numDof > 0:
                found_field = False
                for j in range(c_num_sub_fields):
                    for i_sub_field in range(len_sub_field_array_2[j]):
                        if field == c_sub_field_array[j][i_sub_field]:
                            found_field = True
                    if found_field:
                        c_ISarray[j][c_sub_field_indx[j]] = c_total_field_indx
                        c_sub_field_indx[j] += 1
                        c_total_field_indx += 1
                        break
                if not found_field:
                    raise ValueError("field ID %d not found in fieldsplit array" %field)
                else:
                    break
    for i in range(len_ISarray):
        for j in range(len(ISarray[i])):
            ISarray[i][j] = c_ISarray[i][j]
    total_field_indx = c_total_field_indx
    for i in range(len(sub_field_indx)):
        sub_field_indx[i] = c_sub_field_indx[i]
    return total_field_indx


def assign_dof(section, np.ndarray dofs, goffset, bfk):
    cdef int i
    for i in range(len(dofs)):
        section.setDof(dofs[i]-goffset,1)
        section.setFieldDof(dofs[i]-goffset,bfk,1)

