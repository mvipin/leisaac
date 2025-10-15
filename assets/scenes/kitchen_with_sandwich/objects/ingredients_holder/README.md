# Ingredients Holder

## Overview
This directory contains the ingredients holder asset exported from Fusion 360.

## Expected Structure
```
ingredients_holder/
├── ingredients_holder.usd    # Main USD file (to be added)
├── texture/                  # Textures directory (if needed)
└── README.md                # This file
```

## Fusion 360 Export Instructions

### Export Options (in order of preference):
1. **USD format** (if available) - Direct compatibility
2. **OBJ format** - Good compatibility with USD tools  
3. **STL format** - Fallback option, requires material assignment

### Export Settings:
- **Units**: Meters (to match Isaac Sim)
- **Resolution**: High quality for smooth surfaces
- **Coordinate System**: Y-up (Isaac Sim convention)
- **Include**: All visible components

### Design Requirements:
- **Slots**: 4 slots at 45-degree angles
- **Slot Order**: Left to right: bread_slice_1, bread_slice_2, patty, cheese_slice
- **Dimensions**: Slots sized to fit ingredient objects
- **Material**: Appropriate friction for ingredient sliding
- **Stability**: Base designed to sit securely on counter

## Integration Notes:
- Will be imported into main scene USD file
- Configured as static object (no physics simulation)
- Positioned on kitchen counter within robot reach
- Collision detection enabled for proper ingredient interaction

## Physics Properties:
- **Type**: Static/Fixed object
- **Collision**: Enabled for slots and base
- **Material**: Low friction for easy ingredient sliding
- **Mass**: N/A (static object)
