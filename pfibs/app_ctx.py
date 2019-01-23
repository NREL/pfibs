class AppCtx(object):
    def __init__(self,ctx):
        if not isinstance(ctx,dict):
            raise TypeError("Application context must be of type dict()")
        else:
            self.appctx = ctx
    def content(self):
        return self.appctx
