import numpy as np

ans = np.load("ans.npy")
out = [np.load(f"out_{i}.npy") for i in range(5)]
out = [i/np.linalg.norm(i, axis=1).reshape(-1,1) for i in out]
result = [res.argmax(axis=1) for res in out]
result = np.sum([(res==ans).astype(int) for res in result], axis=0)

result.astype(bool).astype(int).mean()

result_all = np.array(out).mean(axis=0).argmax(axis=1)== ans
print(result_all.astype(int).mean())