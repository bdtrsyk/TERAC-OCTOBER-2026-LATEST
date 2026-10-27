
function save_model(ai::CognitiveAgent, path::String)
    open(path,"w") do io; serialize(io, (version=ai.serialization_version, agent=ai)); end; path
end
function load_model(path::String)
    data=open(deserialize,path); data.version == SERIALIZATION_VERSION || error("unsupported model serialization version")
    assert_network_invariants(data.agent.network); data.agent
end

