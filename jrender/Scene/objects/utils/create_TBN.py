
import jittor as jt

#face_texcoords:[nf,3,3] face_wcoords:[nf,3,3]
#TBN:[nf,3,3] ([nf:T;B;N])
def create_TBN(face_texcoords,face_wcoords):
    e1=face_wcoords[::,0]-face_wcoords[::,1]
    e2=face_wcoords[::,0]-face_wcoords[::,2]
    n=jt.normalize(jt.cross(e1,e2).unsqueeze(1),dim=2)
    u1=(face_texcoords[::,0,0]-face_texcoords[::,1,0])
    v1=(face_texcoords[::,0,1]-face_texcoords[::,1,1])
    u2=(face_texcoords[::,0,0]-face_texcoords[::,2,0])
    v2=(face_texcoords[::,0,1]-face_texcoords[::,2,1])
    denom=jt.array(1/(u1*v2-u2*v1)).float32().unsqueeze(1).unsqueeze(2)
    inverse=jt.array(jt.stack((jt.stack((v2,-v1),dim=1),jt.stack((-u2,u1),dim=1)),dim=1)).float32()
    e=jt.array(jt.stack((e1,e2),dim=1)).float32()
    TB=jt.matmul(inverse,e)
    TB=denom*TB
    T=TB[::,0,::]
    T=T.unsqueeze(1)
    T_n=jt.sum(T*n,dim=2).unsqueeze(1)
    T=T-(T_n*n)
    T=jt.normalize(T,dim=2)
    B=jt.normalize(jt.cross(n,T),dim=2)
    TBN=jt.concat([T,B,n],dim=1)
    return TBN