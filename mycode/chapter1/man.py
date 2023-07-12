class Man:
    def __init__(self,name,tall) -> None:
        self.name = name
        self.tall = tall
        print("initialized")

    def hello(self,toname):
        print(self.name+'say hello to '+toname)

    def goodbye(self,toname):
        print(self.name+"say 886 to "+toname)


man = Man('ljj',180)
man.hello('nobody')
man.goodbye('nobody')