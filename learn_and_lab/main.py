'''
动态规划：青蛙跳阶梯问题
'''
class Solution(object):
    def numWays(self, n):
        """
        :type n: int
        :rtype: int
        """
# f(0)=1，f(1)=1，f(2)=f(0)+f(1)
        a = 1 # f(0)=1
        b = 1 # f(1)=1
        if (n == 0 or n == 1):
            return 1
        i = 2
        while(i!=n+1):
            c = a + b
            a = b # 保存前前一项的
            b = c # 保存前一项  i=2 时 得到 f[2] = 2
            i = i+1
        return c
s = Solution()
ww = s.numWays(4)
# print(ww)


class Solution(object):
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        result = left
        for i in range(left,right + 1):
            result = result & i
        return result
d = Solution()
k = d.rangeBitwiseAnd(5,7)
print(k)


