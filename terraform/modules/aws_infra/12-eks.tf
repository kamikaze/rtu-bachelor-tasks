resource "aws_iam_role" "eks_cluster" {
  count              = var.eks_enabled ? 1 : 0
  name               = "eks-cluster"
  assume_role_policy = <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
POLICY
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster[0].name
}

resource "aws_iam_role_policy_attachment" "eks_service_policy" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
  role       = aws_iam_role.eks_cluster[0].name
}

resource "aws_iam_role_policy_attachment" "eks_vpc_resource_controller" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.eks_cluster[0].name
}

resource "aws_iam_role_policy_attachment" "eks_cluster_cloudwatch_agent_policy_attach" {
  count      = var.eks_enabled ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
  role       = aws_iam_role.eks_cluster[0].name
}

resource "aws_eks_cluster" "eks" {
  count = var.eks_enabled ? 1 : 0
  depends_on = [
    aws_subnet.public_subnet_a,
    aws_subnet.public_subnet_b,
    aws_subnet.public_subnet_c,
    aws_subnet.private_subnet_a,
    aws_subnet.private_subnet_b,
    aws_subnet.private_subnet_c,
    aws_route_table.public_rtb,
    aws_route_table.private_rtb,
    aws_security_group.eks_cluster_sg[0],
    aws_vpc_security_group_ingress_rule.eks_cluster_sg_ipv4_all[0],
    aws_vpc_security_group_ingress_rule.eks_cluster_sg_ipv6_all[0],
    aws_vpc_security_group_egress_rule.eks_cluster_sg_ipv4_all[0],
    aws_vpc_security_group_egress_rule.eks_cluster_sg_ipv6_all[0],
    aws_iam_role_policy_attachment.eks_cluster_policy[0],
    aws_iam_role_policy_attachment.eks_service_policy[0],
    aws_iam_role_policy_attachment.eks_vpc_resource_controller[0]
  ]

  name     = var.eks_cluster_name
  role_arn = aws_iam_role.eks_cluster[0].arn
  version  = var.eks_version

  vpc_config {
    endpoint_private_access = true
    endpoint_public_access  = true

    security_group_ids = [aws_security_group.eks_cluster_sg[0].id]
    subnet_ids = [
      aws_subnet.public_subnet_a.id,
      aws_subnet.public_subnet_b.id,
      aws_subnet.public_subnet_c.id,
      aws_subnet.private_subnet_a.id,
      aws_subnet.private_subnet_b.id,
      aws_subnet.private_subnet_c.id
    ]
  }
}

data "aws_eks_cluster" "eks" {
  count = var.eks_enabled ? 1 : 0
  depends_on = [
    aws_eks_cluster.eks[0]
  ]

  name = aws_eks_cluster.eks[0].name
}

data "aws_eks_cluster_auth" "eks" {
  depends_on = [
    aws_eks_cluster.eks[0]
  ]

  name = aws_eks_cluster.eks[0].name
}
