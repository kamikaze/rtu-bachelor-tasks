resource "aws_iam_role" "workers" {
  count              = var.eks_enabled ? 1 : 0
  name               = "eks-node-group-workers"
  assume_role_policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
POLICY
}

resource "aws_iam_role_policy_attachment" "amazon_eks_worker_node_policy_general" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.workers[0].name
}

resource "aws_iam_role_policy_attachment" "amazon_eks_cni_policy_general" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.workers[0].name
}

resource "aws_iam_role_policy_attachment" "amazon_eks_csi_driver_policy_general" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
  role       = aws_iam_role.workers[0].name
}

resource "aws_iam_role_policy_attachment" "amazon_ec2_container_registry_read_only" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.workers[0].name
}

resource "aws_iam_role_policy_attachment" "amazon_ec2_elb_full_access" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess"
  role       = aws_iam_role.workers[0].name
}

resource "aws_iam_role_policy_attachment" "eks_node_cloudwatch_agent_policy_attach" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
  role       = aws_iam_role.workers[0].name
}

resource "aws_launch_template" "eks_node_lt" {
  depends_on = [
    aws_security_group.eks_node_group_sg[0],
    aws_vpc_security_group_ingress_rule.eks_node_group_sg_ipv4_all[0],
    aws_vpc_security_group_ingress_rule.eks_node_group_sg_ipv6_all[0],
    aws_vpc_security_group_egress_rule.eks_node_group_sg_ipv4_all[0],
    aws_vpc_security_group_egress_rule.eks_node_group_sg_ipv6_all[0],
    aws_security_group.ssh_sg
  ]
  count         = var.eks_enabled ? 1 : 0
  name          = "eks-node-launch-template"
  instance_type = var.eks_node_instance_type
  key_name      = var.admin_ssh_public_key_name

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = 20
      volume_type           = "gp2"
      delete_on_termination = true
    }
  }
  vpc_security_group_ids = [
    aws_security_group.eks_node_group_sg[0].id,
    aws_security_group.ssh_sg.id
  ]

  #  tag_specifications {
  #    resource_type = "instance"
  #    tags = {
  #      "Instance Name" = "${var.eks_cluster_name}-node-${each.value}"
  #       Name = "${var.eks_cluster_name}-node-${each.value}"
  #    }
  #  }
}

resource "aws_eks_node_group" "workers" {
  depends_on = [
    aws_subnet.private_subnet_a,
    aws_subnet.private_subnet_b,
    aws_subnet.private_subnet_c,
    aws_route_table.private_rtb,
    aws_nat_gateway.nat,
    aws_security_group.eks_node_group_sg[0],
    aws_security_group.ssh_sg,
    aws_iam_role_policy_attachment.amazon_eks_worker_node_policy_general[0],
    aws_iam_role_policy_attachment.amazon_eks_cni_policy_general[0],
    aws_iam_role_policy_attachment.amazon_eks_csi_driver_policy_general[0],
    aws_iam_role_policy_attachment.amazon_ec2_elb_full_access[0],
    aws_iam_role_policy_attachment.amazon_ec2_container_registry_read_only[0],
    aws_launch_template.eks_node_lt[0]
  ]

  count           = var.eks_enabled ? 1 : 0
  cluster_name    = aws_eks_cluster.eks[0].name
  node_group_name = "workers"
  node_role_arn   = aws_iam_role.workers[0].arn
  subnet_ids = [
    #    aws_subnet.public_subnet_a.id,
    #    aws_subnet.public_subnet_b.id,
    #    aws_subnet.public_subnet_c.id,
    aws_subnet.private_subnet_a.id,
    aws_subnet.private_subnet_b.id,
    aws_subnet.private_subnet_c.id
  ]
  scaling_config {
    desired_size = 3
    max_size     = 5
    min_size     = 1
  }

  launch_template {
    id      = aws_launch_template.eks_node_lt[0].id
    version = aws_launch_template.eks_node_lt[0].latest_version
  }

  ami_type             = "AL2_ARM_64"
  capacity_type        = "ON_DEMAND"
  force_update_version = false
  release_version      = "${var.eks_version}.0-20240209" # https://github.com/awslabs/amazon-eks-ami/blob/master/CHANGELOG.md
  labels = {
    role = "workers"
  }
  version = aws_eks_cluster.eks[0].version

  tags = {
    Arch         = "arm64"
    HasGPU       = "no"
    Organization = var.organization
    Project      = var.project
    SpotPrice    = "yes"
  }
}