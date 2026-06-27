

using System;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

namespace Characters.Knight
{
    public enum StateCode
    {
        Idle,
        TurnAround,
        Move,
        LookUp,
        LookDown,
        OnGround,
        Fall,
        Jump,
        DoubleJump,
        Coyote,
        Glide,
        Land,
    }

    public class InputData
    {
        public Vector2 movement;
        public Vector2 lookInput;
        public bool onGround;
        public bool onCeiling;
        public bool onLeftWall;
        public bool onRightWall;
        public bool isFacingRight;
        public bool jumpButtonPress;

        public InputData(Vector2 movement, Vector2 lookInput, bool onGround, bool onCeiling, bool onLeftWall, bool  onRightWall, bool isFacingRight, bool jumpButtonPress)
        {
            this.movement = movement;
            this.lookInput = lookInput;
            this.onGround = onGround;
            this.onCeiling = onCeiling;
            this.onLeftWall = onLeftWall;
            this.onRightWall = onRightWall;
            this.isFacingRight = isFacingRight;
            this.jumpButtonPress = jumpButtonPress;
        }
    }

    public class OutputData
    {
        public Vector2 speed;
        public Vector2 angle;
        public int currAnimationCode;
        public int nextAnimationCode;

        public OutputData(Vector2 speed, Vector2 angle, int animationCode)
        {
            this.speed = speed;
            this.angle = angle;
            nextAnimationCode = animationCode;
        }
    }

    public class PropagateData
    {
        public bool isDoubleJumped;
        public Vector2 previousMovement;

        public PropagateData()
        {
            isDoubleJumped = false;
            previousMovement = Vector2.zero;
        }
    }

    public abstract class State
    {
        protected Configuration Config;
        protected OutputData outputs;
        protected float elapsedTime;

        protected State(Configuration config, int animeCode)
        {
            Config = config;
            outputs = new(Vector2.zero, Vector2.zero, animeCode);
        }

        public void Enter(InputData inputs, PropagateData pdata)
        {
            elapsedTime = 0f;
            OnEnter(inputs, pdata);
        }

        protected virtual void OnEnter(InputData inputs, PropagateData pdata) {}

        public void Exit()
        {
            OnExit();
        }
        
        protected virtual void OnExit() {}

        public abstract StateCode Transition(InputData inputs, PropagateData pdata, float dtime);
        public virtual OutputData GetAction()
        {
            return outputs;
        }
    }

    // Horizontal Movement States

    public class IdleState : State
    {
        static readonly int animeCode = Animator.StringToHash("Idle"); 
        
        public IdleState(Configuration config) : base(config, animeCode) {}

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {
            if ( inputs.movement.x < 0 && inputs.isFacingRight || inputs.movement.x > 0 && !inputs.isFacingRight )
            {
                return StateCode.TurnAround;
            }
            else if ( inputs.movement.x != 0 )
            {
                return StateCode.Move;
            }
            else if ( inputs.onGround && inputs.lookInput.y > 0 )
            {
                return StateCode.LookUp;
            }
            else if ( inputs.onGround && inputs.lookInput.y < 0 )
            {
                return StateCode.LookDown;
            }

            return StateCode.Idle;
        }
    }

    public class TurnAroundState : State
    {
        static readonly int animeCode = Animator.StringToHash("TurnAround");
        
        public TurnAroundState(Configuration config) : base(config, animeCode) {}
        
        protected override void OnEnter(InputData inputs, PropagateData pdata)
        {
            if ( inputs.movement.x > 0 )
            {
                outputs.speed.x = Config.turnAroundSpeed;
            }
            else
            {
                outputs.speed.x = -Config.turnAroundSpeed;
            }
        }

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {
            elapsedTime += dtime;
            if ( elapsedTime >= Config.turnAroundDuration )
            {
                if ( inputs.movement.x != 0 )
                {
                    return StateCode.Move;
                }
                else 
                {
                    return StateCode.Idle;
                }
            }

            return StateCode.TurnAround;
        }
    }

    public class MoveState : State
    {
        static readonly int animeCode = Animator.StringToHash("PreMove");
        
        public MoveState(Configuration config) : base(config, animeCode) {}

        protected override void OnEnter(InputData inputs, PropagateData pdata)
        {
            outputs.speed.x = (inputs.movement.x > 0) ? Config.moveSpeed : -Config.moveSpeed ;
        }

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {        
            if ( inputs.movement.x == 0 )
            {
                return StateCode.Idle;
            }
            else if ( outputs.speed.x < 0 && inputs.movement.x > 0 || outputs.speed.x > 0 && inputs.movement.x < 0 )
            {
                // singular occurrence
                return StateCode.TurnAround;
            }

            return StateCode.Move;
        }
    }

    public class LookUpState : State
    {
        static readonly int animeCode = Animator.StringToHash("LookUp");
        private bool lookRelease;
        
        public LookUpState(Configuration config) : base(config, animeCode) {}

        protected override void OnEnter(InputData inputs, PropagateData pdata)
        {
            lookRelease = false;
        }

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {        
            if ( inputs.lookInput.y == 0 )
            {
                lookRelease = true;
            }

            if ( lookRelease )
            {
                elapsedTime += dtime;
                if ( elapsedTime >= Config.postLookUpDuration )
                {
                    return StateCode.Idle;
                }
            }

            return StateCode.LookUp;
        }
    }

    public class LookDownState : State
    {
        static readonly int animeCode = Animator.StringToHash("LookDown");
        private bool lookRelease;

        public LookDownState(Configuration config) : base(config, animeCode) {}

        protected override void OnEnter(InputData inputs, PropagateData pdata)
        {
            lookRelease = false;
        }

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {        
            if ( inputs.lookInput.y == 0 )
            {
                lookRelease = true;
            }

            if ( lookRelease )
            {
                elapsedTime += dtime;
                if ( elapsedTime >= Config.postLookDownDuration )
                {
                    return StateCode.Idle;
                }
            }

            return StateCode.LookDown;
        }
    }


    // Vertical Movement States
    public class OnGroundState : State
    {
        static readonly int animeCode = 0; 
        
        public OnGroundState(Configuration config) : base(config, animeCode) {}

        protected override void OnEnter(InputData inputs, PropagateData pdata)
        {
            pdata.isDoubleJumped = false;
        }

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {
            if ( inputs.movement.y > 0 && inputs.jumpButtonPress )
            {
                return StateCode.Jump;
            }
            else if ( !inputs.onGround )
            {
                return StateCode.Coyote;
            }

            return StateCode.OnGround;
        }
    }

    public class JumpState : State
    {
        static readonly int animeCode = Animator.StringToHash("Jump"); 
        
        public JumpState(Configuration config) : base(config, animeCode) {}

        protected override void OnEnter(InputData inputs, PropagateData pdata)
        {
            outputs.speed.y = Config.jumpSpeed; // for adjustment
        }

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {
            elapsedTime += dtime;
            if ( inputs.movement.y == 0 || inputs.onCeiling )
            {
                return StateCode.Fall;
            } 
            else if ( elapsedTime >= Config.jumpDuration )
            {
                return StateCode.Glide;
            }

            return StateCode.Jump;
        }
    }

    public class DoubleJumpState : State
    {
        static readonly int animeCode = Animator.StringToHash("DoubleJump"); 
        
        public DoubleJumpState(Configuration config) : base(config, animeCode) {}

        protected override void OnEnter(InputData inputs, PropagateData pdata)
        {
            pdata.isDoubleJumped = true;
        }

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {
            elapsedTime += dtime;
            if ( elapsedTime >= Config.doubleJumpPrepareDuration )
            {
                outputs.speed.y = Config.jumpSpeed;
            }
            else
            { 
                outputs.speed.y = 0f;
            }

            if ( inputs.movement.y == 0 || inputs.onCeiling )
            {
                return StateCode.Fall;
            } 
            else if ( elapsedTime >= ( Config.doubleJumpPrepareDuration + Config.doubleJumpDuration ) )
            {
                return StateCode.Glide;
            }

            return StateCode.DoubleJump;
        }
    }

    public class FallState : State
    {
        static readonly int animeCode = Animator.StringToHash("Fall"); 
        
        public FallState(Configuration config) : base(config, animeCode) {}

        protected override void OnEnter(InputData inputs, PropagateData pdata)
        {
            outputs.speed.y = -Config.fallSpeed;
        }

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {
            if ( inputs.onGround )
            {
                return StateCode.Land;
            }
            else if ( inputs.movement.y > 0 && inputs.jumpButtonPress && !pdata.isDoubleJumped )
            {
                return StateCode.DoubleJump;
            }

            return StateCode.Fall;
        }
    }

    public class LandState : State
    {
        static readonly int animeCode = Animator.StringToHash("Land"); 
        
        public LandState(Configuration config) : base(config, animeCode) {}

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {
            elapsedTime += dtime;
            if ( elapsedTime > Config.landDuration )
            {
                return StateCode.OnGround;
            }

            return StateCode.Land;
        }
    }

    public class CoyoteState : State
    {
        static readonly int animeCode = 0; 
        
        public CoyoteState(Configuration config) : base(config, animeCode) {}

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {
            elapsedTime += dtime;
            if ( inputs.onGround )
            {
                return StateCode.OnGround;
            }
            else if ( elapsedTime >= Config.coyoteDuration )
            {
                return StateCode.Fall;
            }
            else if ( inputs.movement.y > 0 && inputs.jumpButtonPress )
            {
                return StateCode.Jump;
            }

            return StateCode.Coyote;
        }
    }

    public class GlideState : State
    {
        static readonly int animeCode = Animator.StringToHash("Glide"); 
        
        public GlideState(Configuration config) : base(config, animeCode) {}

        public override StateCode Transition(InputData inputs, PropagateData pdata, float dtime)
        {
            elapsedTime += dtime;
            
            if ( elapsedTime < Config.preGlideDuration )
            {
                outputs.speed.y = Mathf.Lerp(Config.jumpSpeed, Config.glideSpeed, elapsedTime / Config.glideDuration);
            }            
            else if ( elapsedTime < ( Config.preGlideDuration + Config.glideDuration ) )
            {
                outputs.speed.y = Config.glideSpeed;
            }
            else
            {
                outputs.speed.y = Mathf.Lerp(Config.glideSpeed, -Config.fallSpeed, elapsedTime / Config.glideDuration);
            }

            if ( inputs.onGround )
            {
                return StateCode.OnGround;
            }
            else if ( elapsedTime >= ( Config.preGlideDuration + Config.glideDuration + Config.postGlideDuration ) )
            {
                return StateCode.Fall;
            }
            else if ( inputs.movement.y > 0 && inputs.jumpButtonPress && !pdata.isDoubleJumped )
            {
                return StateCode.DoubleJump;
            }

            return StateCode.Glide;
        }
    }

    public class StateMachine
    {
        private Dictionary<StateCode, State> _stateTable = new();
        private StateCode _currentStateCode;
        private State _currentState;
        private PropagateData _propagateData = new();

        public void RegisterState(StateCode code, State state)
        {
            _stateTable.Add(code, state);    
        }

        public void SetState(StateCode code)
        {
            _currentStateCode = code;
            _currentState = _stateTable[code];  
        }

        public void Transition(InputData inputs, float dtime)
        {
            StateCode nextStateCode = _currentState.Transition(inputs, _propagateData, dtime);

            if ( nextStateCode !=  _currentStateCode )
            {
                _currentStateCode = nextStateCode;
                _currentState?.Exit();
                _currentState = _stateTable[nextStateCode];
                _currentState.Enter(inputs, _propagateData);   
            }   
        }

        public OutputData GetAction()
        {
            return _currentState.GetAction();
        }
    }

    public class HierarchyStateMachine
    {
        private StateMachine horizontalMovementStateMachine;

        private StateMachine verticalMovementStateMachine;

        private OutputData outputs;

        public HierarchyStateMachine(Configuration config)
        {
            horizontalMovementStateMachine = new();
            horizontalMovementStateMachine.RegisterState(StateCode.Idle, new IdleState(config));
            horizontalMovementStateMachine.RegisterState(StateCode.Move, new MoveState(config));
            horizontalMovementStateMachine.RegisterState(StateCode.TurnAround, new TurnAroundState(config));
            horizontalMovementStateMachine.RegisterState(StateCode.LookUp, new LookUpState(config));
            horizontalMovementStateMachine.RegisterState(StateCode.LookDown, new LookDownState(config));
            horizontalMovementStateMachine.SetState(StateCode.Idle);

            verticalMovementStateMachine = new();
            verticalMovementStateMachine.RegisterState(StateCode.OnGround, new OnGroundState(config));
            verticalMovementStateMachine.RegisterState(StateCode.Jump, new JumpState(config));
            verticalMovementStateMachine.RegisterState(StateCode.DoubleJump, new DoubleJumpState(config));
            verticalMovementStateMachine.RegisterState(StateCode.Fall, new FallState(config));
            verticalMovementStateMachine.RegisterState(StateCode.Coyote, new CoyoteState(config));
            verticalMovementStateMachine.RegisterState(StateCode.Glide, new GlideState(config));
            verticalMovementStateMachine.RegisterState(StateCode.Land, new LandState(config));
            verticalMovementStateMachine.SetState(StateCode.OnGround);

            outputs = new(Vector2.zero, Vector2.zero, 0);
        }

        public void Transition(InputData inputs, float dtime)
        {
            horizontalMovementStateMachine.Transition(inputs, dtime);
            verticalMovementStateMachine.Transition(inputs, dtime);
        }

        public OutputData GetAction()
        {
            OutputData hout = horizontalMovementStateMachine.GetAction();
            OutputData vout = verticalMovementStateMachine.GetAction();

            outputs.currAnimationCode = outputs.nextAnimationCode;

            if ( vout.nextAnimationCode != 0 )
            {
                outputs.nextAnimationCode = vout.nextAnimationCode;
            }
            else
            {
                outputs.nextAnimationCode = hout.nextAnimationCode;
            }

            outputs.speed.x = hout.speed.x;
            outputs.speed.y = vout.speed.y;

            return outputs;
        }
    }
}