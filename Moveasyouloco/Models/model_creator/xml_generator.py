import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse

class LocoMujocoLimiter:
    def __init__(self, template_xml_path):
        self.tree = ET.parse(template_xml_path)
        self.root = self.tree.getroot()

        # O(1) lookup dictionary for all <joint> tags.
        # Note: This naturally ignores <freejoint name="root"/> which is what we want.
        self.joints = {joint.get('name'): joint for joint in self.root.iter('joint') if joint.get('name')}
        print(f"✅ Loaded template XML with {len(self.joints)} hinge/actuated joints.")

    def calculate_limits(self, npz_path, padding_rad=0.087):
        """
        Extract min/max joint angles from loco-mujoco .npz and add padding.
        0.087 radians is roughly 5 degrees of padding.
        """
        with np.load(npz_path, allow_pickle=True) as data:
            # 'qpos' is the joint position array in MuJoCo
            qpos = data['qpos']
            joint_names = data['joint_names']

            if isinstance(joint_names[0], bytes):
                joint_names = [n.decode('utf-8') for n in joint_names]

        # For each joint, compute its qpos indices and extract limits
        limits_dict = {}
        qpos_idx = 0
        
        for joint_name in joint_names:
            # Skip the root free joint (7 qpos indices: 3 pos + 4 quat)
            if joint_name == "root":
                qpos_idx += 7
                continue
            
            # Each hinge/slide joint has 1 qpos index
            min_angle = np.min(qpos[:, qpos_idx])
            max_angle = np.max(qpos[:, qpos_idx])
            
            limits_dict[joint_name] = {
                'min': min_angle - padding_rad,
                'max': max_angle + padding_rad
            }
            
            qpos_idx += 1
        
        return limits_dict

    def apply_limits_to_xml(self, limits_dict):
        """Update the range attribute for each joint in the XML."""
        updated = 0
        skipped = 0

        print(f"\n🔧 UPDATING JOINT KINEMATIC LIMITS")
        print(f"{'Joint Name':<25} | {'Old Range':<25} | {'New Range (Radians)':<25}")
        print("-" * 80)

        for joint_name, limits in limits_dict.items():
            if joint_name in self.joints:
                joint_elem = self.joints[joint_name]
                old_range = joint_elem.get('range', 'None')

                # Format exactly to 4 decimal places
                new_range_str = f"{limits['min']:.4f} {limits['max']:.4f}"

                # Update the XML
                joint_elem.set('range', new_range_str)
                # Ensure limited is explicitly true on the joint itself, just in case
                joint_elem.set('limited', 'true')

                print(f"{joint_name:<25} | [{old_range:<23}] | [{new_range_str}]")
                updated += 1
            else:
                skipped += 1

        print(f"\n✅ Updated: {updated} | ⏭️ Skipped: {skipped} (Not found in XML)")

    def save(self, output_path):
        """Save the modified XML."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(ET, 'indent'):
            ET.indent(self.tree, space="  ", level=0)

        self.tree.write(out_path, encoding='utf-8', xml_declaration=True)
        print(f"💾 Saved strictly-limited skeleton to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Limit Loco-Mujoco Skeleton from NPZ data.')
    parser.add_argument('--template', type=str, default='Moveasyouloco/Models/skeleton/skeleton_torque.xml') #hier ook nog de goede path
    parser.add_argument('--data', type=str, required=True, help='Path to the .npz motion file')
    parser.add_argument('--output', type=str, default='skeleton_limited.xml') #misschien dit nog ff aanpassen naar een logischere path
    parser.add_argument('--padding', type=float, default=0.087, help='Padding in radians (default: 5 degrees)')
    args = parser.parse_args()

    limiter = LocoMujocoLimiter(args.template)
    limits = limiter.calculate_limits(args.data, padding_rad=args.padding)
    limiter.apply_limits_to_xml(limits)
    limiter.save(args.output)

if __name__ == "__main__":
    main()