from enum import Enum


functions = [
    {
        "name": "set_temperature",
        "description": "Set the temperature in a specified zone of the car.",
        "parameters": {
            "type": "object",
            "properties": {
                "area": {
                    "type": "array[string]",
                    "enum": ["driver",'front-passenger','rear-right','rear-left'],
                    "description": "The zone where the temperature will be adjusted.",
                    "default": "all"
                },
                "temperature": {
                    "type": "number",
                    "description": "The target temperature for the specified zone in degrees Celsius.",
                    "lower_bound": 1,
                    "upper_bound": 80
                },
                "unit": {
                    "type": "string",
                    "description": "The unit of the temperature value.",
                    "enum": ["Celsius", "Fahrenheit"],
                    "default": "Celsius"
                }
            },
            "required": ["temperature"],
            "optional": ["area","unit"]
        }
    },
    {
        "name": "adjust_fan_speed",
        "description": "Adjust the fan speed of the car's climate control system.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {
                    "type": "string",
                    "description": "The desired fan speed level",
                    "enum": ['increase','decrease'],
                    "default": 'increase',
                },
                'area': {
                    "type": "array[string]",
                    "description": "The area to adjust the fan speed for.",
                    "enum": ["driver",'front-passenger','rear-right','rear-left'],
                    "default": "all" 
                },
            },
            "required": [
                "speed"
            ],
            "optional": ["area"]
        }
    },
    {
        'name':'set_fan_speed',
        'description':'Set the fan speed of the car\'s climate control system.',
        'parameters':{
            'type':'object',
            'properties':{
                'speed':{
                    'type':'string',
                    'description':'The desired fan speed level',
                    'enum': ['LOW','MEDIUM','HIGH'],
                    'default': 'MEDIUM',
                },
                'area':{
                    'type':'array[string]',
                    'description':'The area to adjust the fan speed for.',
                    'enum': ["driver",'front-passenger','rear-right','rear-left'],
                    'default': 'all'
                }
            },
            'required': ['speed'],
            'optional': ['area']
        },
    },
    {
        "name": "adjust_temperature",
        "description": "Adjust the temperature in a specified zone of the car.",
        "parameters": {
            "type": "object",
            "properties": {
                "area": {
                    "type": "array[string]",
                    "enum": ["driver",'front-passenger','rear-right','rear-left'],
                    "description": "The zone where the temperature will be adjusted.",
                    "default": "all"
                },
                "action": {
                    "type": "string",
                    "description": "The action to perform on the temperature (e.g., 'increase', 'decrease').",
                    "enum": ["increase", "decrease"],
                    "default": "increase"
                },
            },
            "required": ["action"],
            "optional": ["area"]
        }
    },
    {
        "name": "adjust_seat",
        "description": "Adjust a seat's position in the car.",
        "parameters": {
            "type": "object",
            "properties": {
                "seat_type": {
                    "type": "string",
                    "enum": ["driver", "front-passenger", "rear_right", "rear_left"],
                    "description": "The type of seat to adjust.",
                    "default": "driver"
                },
                "position": {
                    "type": "string",
                    "description": "The desired position of the seat (e.g., 'forward', 'backward', 'up', 'down').",
                    "enum": ["forward", "backward", "up", "down","tilt-forward","tilt-backward"],
                },
            },
            "required": ["position"],
            "optional": ["seat_type"],
        }
    },
    {
        "name": "control_window",
        "description": "Control the car window's position.",
        "parameters": {
            "type": "object",
            "properties": {
                "window_position": {
                    "type": "string",
                    "description": "The desired position of the window (e.g., 'up', 'down').",
                    "enum": ["open", "close"]
                },
                "window_location": {
                    "type": "string",
                    "description": "The location of the window (e.g., 'driver', 'front-passenger', 'rear_right', 'rear_left').",
                    "default": "all",
                    "enum": ["driver", "front-passenger", "rear_right", "rear_left"]
                }
            },
            "required": ["window_position"],
            "optional": ["window_location"]
        }
    },
    {
        "name": "adjust_wiper_speed",
        "description": "Adjust the windshield wipers.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {
                    "type": "string",
                    "description": "The speed of the wipers (e.g., 1 for low, 2 for medium, 3 for high).",
                    "enum": ['INCREASE','DECREASE'],
                }
            },
            "required": [
                "speed"
            ],
            "optional": []
        }
    },
    {
        "name": "set_wiper_speed",
        "description": "Activate the windshield wipers.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {
                    "type": "integer",
                    "description": "The speed of the wipers (e.g., 1 for low, 2 for medium, 3 for high).",
                    "enum": ['HIGH','MEDIUM','LOW'],
                }
            },
            "required": [
                "speed"
            ],
            "optional": []
        }
    },
    {
        "name": "activate_defroster",
        "description": "Activate the defroster for windows and windshield.",
        "parameters": {
            "type": "object",
            "properties": {
                "defroster_zone": {
                    "type": "string",
                    "description": "The zone to defrost (e.g., 'front', 'rear', 'all').",
                    "default": "all",
                    "enum": ["front", "rear", "all"]
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Duration in minutes for which the defroster should be active.",
                    "default": 10,
                    "lower_bound": 1,
                    "upper_bound": 30
                }
            },
            "required": [],
            "optional": [
                "duration_minutes", "defroster_zone"
            ]
        }
    },
    {
        "name": "start_engine",
        "description": "Start the car's engine remotely.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "The method to start the engine (e.g., 'remote', 'keyless', 'keyed').",
                    "default": "keyless",
                    "enum": ["remote", "keyless", "keyed"]
                }
            },
            "optional": [
                "method"
            ],
            "required": []
        }
    },
    {
        "name": "lock_doors",
        "description": "Lock or unlock the car doors.",
        "parameters": {
            "type": "object",
            "properties": {
                "lock_state": {
                    "type": "string",
                    "description": "Set to 'lock' to lock the doors, 'unlock' to unlock.",
                    "default": "lock",
                    "enum": ["lock", "unlock"]
                }
            },
            "required": [
                "lock_state"
            ],
            "optional": []
        }
    },
    {
        "name": "play_music",
        "description": "Control the music player in the car.",
        "parameters": {
            "type": "object",
            "properties": {
                "track": {
                    "type": "string",
                    "description": "The track name to play.",
                    "default": "random"
                },
                "volume": {
                    "type": "integer",
                    "description": "Volume level from 1 (low) to 10 (high).",
                    "default": 5,
                    "lower_bound": 1,
                    "upper_bound": 10
                }
            },
            "required": [],
            "optional": [
                "volume","track"
            ]
        }
    },
    {
        "name": "toggle_headlights",
        "description": "Turn the headlights on or off.",
        "parameters": {
            "type": "object",
            "properties": {
                "light_state": {
                    "type": "string",
                    "description": "Set to 'on' to turn the headlights on, 'off' to turn them off.",
                    "enum": ["on", "off"]
                }
            },
            "required": [
                "light_state"
            ]
        }
    },
    {
        "name": "set_navigation_destination",
        "description": "Set a destination in the car's navigation system.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "The address or location to navigate to."
                }
            },
            "required": [
                "destination"
            ],
            "optional": []
        }
    },
    {
        "name": "control_ambient_lighting",
        "description": "Adjust the color and intensity of the interior ambient lighting.",
        "parameters": {
            "type": "object",
            "properties": {
                "color": {
                    "type": "string",
                    "description": "The color of the ambient lighting.",
                    "enum":["warm", "red","blue","dark","white"]
                },
                "intensity": {
                    "type": "integer",
                    "description": "The intensity level of the lighting, from 1 (low) to 10 (high).",
                    "default": 5,
                    "lower_bound": 1,
                    "upper_bound": 10
                }
            },
            "required": [
                "color"
            ],
            "optional": [
                "intensity"
            ]
        }
    },
    {
        "name": "set_cruise_control",
        "description": "Activate and set the speed for cruise control.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {
                    "type": "integer",
                    "description": "The cruise control speed in km/h.",
                    "lower_bound": 10,
                    "upper_bound": 150
                }
            },
            "required": [
                "speed"
            ],
            "optional": []
        }
    },
    {
        "name": "check_battery_health",
        "description": "Provide the current status and health of the car's battery.",
        "parameters": {
            "type": "object",
            "properties": {
                "include_history": {
                    "type": "boolean",
                    "description": "Whether to include historical health data.",
                    "default": False
                }
            },
            "optional": [
                "include_history"
            ],
            "required": []
        }
    },
    {
        "name": "toggle_sport_mode",
        "description": "Toggle the car's sport mode setting.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Set to true to enable sport mode, false to disable.",
                    "enum": ["activate", "deactivate"]
                }
            },
            "required": [
                "action"
            ],
            "optional": []
        }
    }
]


class ErrorType(Enum):
    INVALID_FUNCTION = "InvalidFunction"
    MISSING_PARAMETER = "MissingParameter"
    INCORRECT_PARAMETER_VALUE = "IncorrectParameterValue"
    HALLUCINATED_PARAMETER = "HallucinatedParameter"
    HALLUCINATED_PARAMETER_VALUE = "HallucinatedParameterValue"
    HALLUCINATED_FUNCTION = "HallucinatedFunction"
    INCORRECT_PARAMETER_TYPE_ARRAY = "IncorrectParameterTypeArray"
    MISSING_ARRAY_ELEMENT = "MissingArrayElement"
    HALLUCINATED_ARRAY_ELEMENT = "HallucinatedArrayElement"
    INCORRECT_ARRAY_ELEMENT = "IncorrectArrayElement"
    