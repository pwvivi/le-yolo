class MBConv(nn.Module):
    def __init__(self, inc, ouc, shortcut=True, e=4, dropout=0.1) -> None:
        super().__init__()
        midc = inc * e
        self.conv_pw_1 = Conv(inc, midc, 1)
        self.conv_dw_1 = Conv(midc, midc, 3, g=midc)
        self.effective_se = EffectiveSEModule(midc)
        self.conv1 = Conv(midc, ouc, 1, act=False)
        self.dropout = nn.Dropout2d(p=dropout)
        self.add = shortcut and inc == ouc
    
    def forward(self, x):
        return x + self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x))))) if self.add else self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))


class C2f_ESEMB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MBConv(self.c, self.c, shortcut) for _ in range(n))
