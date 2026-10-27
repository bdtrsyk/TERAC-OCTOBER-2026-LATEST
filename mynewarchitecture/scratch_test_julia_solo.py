from tcai_bridge import JuliaBridge

j = JuliaBridge(scale="tiny")
resp = j.ask("does sun cause heat?")
print("Julia response for 'does sun cause heat?':", resp)

# Now teach it
t_resp = j.teach("sun causes light. light causes heat.", "sun causes heat")
print("After teach:", t_resp)

resp2 = j.ask("does sun cause heat?")
print("Julia response after teach:", resp2)

j.close()
