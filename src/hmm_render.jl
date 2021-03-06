using PyPlot
using SMC

function render_hmm!(hmm::HiddenMarkovModel)
    # renders a HMM in the current figure
    # example usage:
    # plt[:figure]()
    # render!(hmm)
    # plt[:savefig]("hmm.png")

    num_states = hmm.num_states
    num_obs = hmm.num_obs

    # three subplots left to right:
    # 1. the prior (column vector)
    # 2. the transition matrix (square matrix)
    # 3. the observation matrix (rectangle)
    # the height is always num_states
    width_ratios = [num_states, num_obs]
    gs = matplotlib[:gridspec][:GridSpec](1, 2, width_ratios=width_ratios)
    trans_ax = plt[:subplot](gs[1])
    obs_ax = plt[:subplot](gs[2])
    prior_mat = Array{Float64,2}(num_states,1)
    prior_mat[:,1] = hmm.initial_state_prior
    trans_ax[:imshow](hmm.transition_model, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    obs_ax[:imshow](hmm.observation_model, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    trans_ax[:set_title]("transition model")
    obs_ax[:set_title]("observation model")
    for ax in [trans_ax, obs_ax]
        ax[:set_yticks](0:num_states-1)
        ax[:set_yticklabels](1:num_states)
    end

    trans_ax[:set_ylabel]("previous state", fontsize=12)
    trans_ax[:set_xlabel]("next state", fontsize=12)
    trans_ax[:set_xticks](0:num_states-1)
    trans_ax[:set_xticklabels](1:num_states)
    trans_ax[:set_xlim]([-0.5, num_states - 0.5])
    trans_ax[:set_ylim]([-0.5, num_states- 0.5])
    trans_ax[:set_xticks]((1:num_states-1) - 0.5, minor=true)
    trans_ax[:set_yticks]((1:num_states-1) - 0.5, minor=true)

    obs_ax[:set_ylabel]("state", fontsize=12)
    obs_ax[:set_xlabel]("observation", fontsize=12)
    obs_ax[:set_xticks](0:num_obs-1)
    obs_ax[:set_xticklabels](1:num_obs)
    obs_ax[:set_xlim]([-0.5, num_obs - 0.5])
    obs_ax[:set_ylim]([-0.5, num_states- 0.5])
    obs_ax[:set_xticks]((1:num_obs-1) - 0.5, minor=true)
    obs_ax[:set_yticks]((1:num_states-1) - 0.5, minor=true)

    plt[:tight_layout]()
end

function render_hmm_states!(hmm::HiddenMarkovModel, states::Array{Int,1},
                            grid::Bool, xticks, yticks)
    if maximum(states) > hmm.num_states || minimum(states) < 1
        error("bad states")
    end
    mat = zeros(hmm.num_states, length(states))
    for i=1:length(states)
        mat[states[i], i] = 1.0
    end
    ax = plt[:gca]()

    # white will be 0.0, and black will be 1.0 (to save ink)
    ax[:imshow](1.0 - mat, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)

    ax[:set_yticks](yticks-1)
    ax[:set_xticks](xticks-1)
    ax[:set_yticklabels](yticks)
    ax[:set_xticklabels](xticks)

    ax[:set_xlim]([-0.5, length(states) - 0.25])
    ax[:set_ylim]([-0.5, hmm.num_states - 0.25])

    if grid
        # set grid lines using minor ticks
        ax[:set_xticks]((1:length(states)-1) - 0.5, minor=true);
        ax[:set_yticks]((1:hmm.num_states-1) - 0.5, minor=true);
        ax[:grid](which="minor", color="orange", linewidth=1)
    end
end

function render_hmm_posterior_marginals!(hmm::HiddenMarkovModel, 
                                         marginals::Array{Float64,2},
                                         grid::Bool, xticks, yticks)
    if size(marginals)[1] != hmm.num_states
        error("bad states")
    end
    num_steps = size(marginals)[2]
    ax = plt[:gca]()

    # white will be 0.0, and black will be 1.0 (to save ink)
    ax[:imshow](1.0 - marginals, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)

    ax[:set_yticks](yticks-1)
    ax[:set_xticks](xticks-1)
    ax[:set_yticklabels](yticks)
    ax[:set_xticklabels](xticks)
    ax[:set_xlim]([-0.5, num_steps - 0.25])
    ax[:set_ylim]([-0.5, hmm.num_states - 0.25])

    # set grid lines using minor ticks
    if grid
        ax[:set_xticks]((1:num_steps-1) - 0.5, minor=true);
        ax[:set_yticks]((1:hmm.num_states-1) - 0.5, minor=true);
        ax[:grid](which="minor", color="orange", linewidth=1)
    end
end


function render_hmm_observations!(hmm::HiddenMarkovModel, 
                                  observations::Array{Int,1})
    if maximum(observations) > hmm.num_obs || minimum(observations) < 1
        error("bad observations")
    end
    mat = zeros(hmm.num_obs, length(observations))
    for i=1:length(observations)
        mat[observations[i], i] = 1.0
    end
    ax = plt[:gca]()
    ax[:imshow](mat, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
    ax[:set_ylabel]("observation", fontsize=12)
    ax[:set_xlabel]("time step", fontsize=12)
    ax[:set_yticks](0:2:hmm.num_obs-1)
    ax[:set_yticklabels](1:2:hmm.num_obs)
    ax[:set_xticks](0:2:length(observations))
    ax[:set_xticklabels](1:2:length(observations))
    ax[:set_xlim]([-0.5, length(observations) - 0.5])
    ax[:set_ylim]([-0.5, hmm.num_obs - 0.5])

    # set grid lines using minor ticks
    ax[:set_xticks]((1:length(observations)-1) - 0.5, minor=true);
    ax[:set_yticks]((1:hmm.num_obs-1) - 0.5, minor=true);
end

